#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

/* --- Pairwise Repulsion --- */

typedef struct {
    int from;
    int to;
    int n;
    const double* pos;
    const double* mass;
    const double* sizes;
    double* forces;
    double coefficient;
    int adjust_sizes;
} RepulsionThreadData;

static void* repulsion_worker(void* arg) {
    RepulsionThreadData* d = (RepulsionThreadData*)arg;
    int n = d->n;
    int adjust_sizes = d->adjust_sizes;
    double coef = d->coefficient;
    const double* pos = d->pos;
    const double* mass = d->mass;
    const double* sizes = d->sizes;
    double* forces = d->forces;

    for (int i = d->from; i < d->to; i++) {
        double px = pos[2 * i];
        double py = pos[2 * i + 1];
        double mi = mass[i];
        double si = (adjust_sizes && sizes) ? sizes[i] : 0.0;
        double fx = 0.0;
        double fy = 0.0;

        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            double x_dist = px - pos[2 * j];
            double y_dist = py - pos[2 * j + 1];
            double dist = sqrt(x_dist * x_dist + y_dist * y_dist);

            if (adjust_sizes && sizes) {
                double dist_c = dist - si - sizes[j];
                if (dist_c > 0) {
                    double factor = coef * mi * mass[j] / (dist_c * dist_c);
                    fx += x_dist * factor;
                    fy += y_dist * factor;
                } else if (dist_c < 0) {
                    double factor = 100.0 * coef * mi * mass[j];
                    fx += x_dist * factor;
                    fy += y_dist * factor;
                }
            } else {
                if (dist > 0) {
                    double factor = coef * mi * mass[j] / (dist * dist);
                    fx += x_dist * factor;
                    fy += y_dist * factor;
                }
            }
        }
        forces[2 * i] += fx;
        forces[2 * i + 1] += fy;
    }
    return NULL;
}

void c_repulsion_pairwise(
    const double* pos,
    const double* mass,
    const double* sizes,
    double* forces,
    int n,
    double coefficient,
    int adjust_sizes,
    int num_threads
) {
    if (num_threads <= 1 || n < 100) {
        RepulsionThreadData data = {0, n, n, pos, mass, sizes, forces, coefficient, adjust_sizes};
        repulsion_worker(&data);
        return;
    }

    if (num_threads > 32) num_threads = 32;
    pthread_t threads[32];
    RepulsionThreadData tdata[32];
    int chunk = (n + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; t++) {
        tdata[t].from = t * chunk;
        tdata[t].to = MIN(n, (t + 1) * chunk);
        tdata[t].n = n;
        tdata[t].pos = pos;
        tdata[t].mass = mass;
        tdata[t].sizes = sizes;
        tdata[t].forces = forces;
        tdata[t].coefficient = coefficient;
        tdata[t].adjust_sizes = adjust_sizes;
        if (tdata[t].from < n) {
            pthread_create(&threads[t], NULL, repulsion_worker, &tdata[t]);
        }
    }

    for (int t = 0; t < num_threads; t++) {
        if (tdata[t].from < n) {
            pthread_join(threads[t], NULL);
        }
    }
}

/* --- Barnes-Hut Linear Quadtree --- */

typedef struct {
    double mass;
    double cx;
    double cy;
    double size;
    int children[4];
    int node_id; // -1 if internal, >=0 if leaf
} BHNode;

typedef struct {
    BHNode* nodes;
    int capacity;
    int count;
} BHTree;

static int bhtree_alloc_node(BHTree* tree) {
    if (tree->count >= tree->capacity) {
        tree->capacity *= 2;
        tree->nodes = (BHNode*)realloc(tree->nodes, sizeof(BHNode) * tree->capacity);
    }
    int idx = tree->count++;
    BHNode* n = &tree->nodes[idx];
    n->mass = 0.0;
    n->cx = 0.0;
    n->cy = 0.0;
    n->size = 0.0;
    n->children[0] = -1;
    n->children[1] = -1;
    n->children[2] = -1;
    n->children[3] = -1;
    n->node_id = -1;
    return idx;
}

static int build_bh_recursive(
    BHTree* tree,
    int* node_ids,
    int num_nodes,
    const double* pos,
    const double* mass
) {
    int node_idx = bhtree_alloc_node(tree);
    if (num_nodes == 0) return -1;

    if (num_nodes == 1) {
        int id = node_ids[0];
        BHNode* n = &tree->nodes[node_idx];
        n->mass = mass[id];
        n->cx = pos[2 * id];
        n->cy = pos[2 * id + 1];
        n->size = 0.0;
        n->node_id = id;
        return node_idx;
    }

    double total_mass = 0.0;
    double sum_x = 0.0;
    double sum_y = 0.0;
    for (int i = 0; i < num_nodes; i++) {
        int id = node_ids[i];
        double m = mass[id];
        total_mass += m;
        sum_x += pos[2 * id] * m;
        sum_y += pos[2 * id + 1] * m;
    }

    double cx = (total_mass > 0) ? (sum_x / total_mass) : 0.0;
    double cy = (total_mass > 0) ? (sum_y / total_mass) : 0.0;

    double max_size = 0.0;
    for (int i = 0; i < num_nodes; i++) {
        int id = node_ids[i];
        double dx = pos[2 * id] - cx;
        double dy = pos[2 * id + 1] - cy;
        double d = sqrt(dx * dx + dy * dy);
        if (2.0 * d > max_size) max_size = 2.0 * d;
    }

    int* q_nodes[4];
    int q_counts[4] = {0, 0, 0, 0};
    for (int k = 0; k < 4; k++) {
        q_nodes[k] = (int*)malloc(sizeof(int) * num_nodes);
    }

    for (int i = 0; i < num_nodes; i++) {
        int id = node_ids[i];
        double px = pos[2 * id];
        double py = pos[2 * id + 1];
        int quad;
        if (px < cx) {
            quad = (py < cy) ? 0 : 1;
        } else {
            quad = (py < cy) ? 2 : 3;
        }
        q_nodes[quad][q_counts[quad]++] = id;
    }

    int all_in_one = -1;
    for (int k = 0; k < 4; k++) {
        if (q_counts[k] == num_nodes) {
            all_in_one = k;
            break;
        }
    }

    int children[4] = {-1, -1, -1, -1};
    if (all_in_one != -1) {
        for (int i = 0; i < num_nodes && i < 4; i++) {
            int one_id = node_ids[i];
            int leaf_idx = bhtree_alloc_node(tree);
            BHNode* ln = &tree->nodes[leaf_idx];
            ln->mass = mass[one_id];
            ln->cx = pos[2 * one_id];
            ln->cy = pos[2 * one_id + 1];
            ln->size = 0.0;
            ln->node_id = one_id;
            children[i] = leaf_idx;
        }
    } else {
        for (int k = 0; k < 4; k++) {
            if (q_counts[k] > 0) {
                children[k] = build_bh_recursive(tree, q_nodes[k], q_counts[k], pos, mass);
            }
        }
    }

    for (int k = 0; k < 4; k++) {
        free(q_nodes[k]);
    }

    BHNode* n = &tree->nodes[node_idx];
    n->mass = total_mass;
    n->cx = cx;
    n->cy = cy;
    n->size = max_size;
    n->node_id = -1;
    for (int k = 0; k < 4; k++) {
        n->children[k] = children[k];
    }

    return node_idx;
}

typedef struct {
    int from;
    int to;
    int n;
    const double* pos;
    const double* mass;
    const double* sizes;
    double* forces;
    const BHNode* bh_nodes;
    double coefficient;
    double theta;
    int adjust_sizes;
} BHThreadData;

static void* bh_repulsion_worker(void* arg) {
    BHThreadData* d = (BHThreadData*)arg;
    const double* pos = d->pos;
    const double* mass = d->mass;
    const double* sizes = d->sizes;
    double* forces = d->forces;
    const BHNode* bh_nodes = d->bh_nodes;
    double coef = d->coefficient;
    double theta = d->theta;
    int adjust_sizes = d->adjust_sizes;

    int stack[128];

    for (int i = d->from; i < d->to; i++) {
        double px = pos[2 * i];
        double py = pos[2 * i + 1];
        double mi = mass[i];
        double si = (adjust_sizes && sizes) ? sizes[i] : 0.0;
        double fx = 0.0;
        double fy = 0.0;

        int top = 0;
        stack[top++] = 0;

        while (top > 0) {
            int curr = stack[--top];
            const BHNode* bn = &bh_nodes[curr];

            if (bn->node_id >= 0) {
                int j = bn->node_id;
                if (i != j) {
                    double x_dist = px - pos[2 * j];
                    double y_dist = py - pos[2 * j + 1];
                    double dist = sqrt(x_dist * x_dist + y_dist * y_dist);

                    if (adjust_sizes && sizes) {
                        double dist_c = dist - si - sizes[j];
                        if (dist_c > 0) {
                            double factor = coef * mi * mass[j] / (dist_c * dist_c);
                            fx += x_dist * factor;
                            fy += y_dist * factor;
                        } else if (dist_c < 0) {
                            double factor = 100.0 * coef * mi * mass[j];
                            fx += x_dist * factor;
                            fy += y_dist * factor;
                        }
                    } else {
                        if (dist > 0) {
                            double factor = coef * mi * mass[j] / (dist * dist);
                            fx += x_dist * factor;
                            fy += y_dist * factor;
                        }
                    }
                }
            } else {
                double x_dist = px - bn->cx;
                double y_dist = py - bn->cy;
                double dist = sqrt(x_dist * x_dist + y_dist * y_dist);

                if (dist * theta > bn->size) {
                    if (dist > 0) {
                        double factor = coef * mi * bn->mass / (dist * dist);
                        fx += x_dist * factor;
                        fy += y_dist * factor;
                    }
                } else {
                    for (int k = 0; k < 4; k++) {
                        if (bn->children[k] >= 0 && top < 127) {
                            stack[top++] = bn->children[k];
                        }
                    }
                }
            }
        }
        forces[2 * i] += fx;
        forces[2 * i + 1] += fy;
    }
    return NULL;
}

void c_repulsion_barnes_hut(
    const double* pos,
    const double* mass,
    const double* sizes,
    double* forces,
    int n,
    double coefficient,
    double theta,
    int adjust_sizes,
    int num_threads
) {
    if (n <= 1) return;

    BHTree tree;
    tree.capacity = n * 4;
    tree.count = 0;
    tree.nodes = (BHNode*)malloc(sizeof(BHNode) * tree.capacity);

    int* all_ids = (int*)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) all_ids[i] = i;

    build_bh_recursive(&tree, all_ids, n, pos, mass);
    free(all_ids);

    if (num_threads <= 1 || n < 100) {
        BHThreadData data = {0, n, n, pos, mass, sizes, forces, tree.nodes, coefficient, theta, adjust_sizes};
        bh_repulsion_worker(&data);
    } else {
        if (num_threads > 32) num_threads = 32;
        pthread_t threads[32];
        BHThreadData tdata[32];
        int chunk = (n + num_threads - 1) / num_threads;

        for (int t = 0; t < num_threads; t++) {
            tdata[t].from = t * chunk;
            tdata[t].to = MIN(n, (t + 1) * chunk);
            tdata[t].n = n;
            tdata[t].pos = pos;
            tdata[t].mass = mass;
            tdata[t].sizes = sizes;
            tdata[t].forces = forces;
            tdata[t].bh_nodes = tree.nodes;
            tdata[t].coefficient = coefficient;
            tdata[t].theta = theta;
            tdata[t].adjust_sizes = adjust_sizes;
            if (tdata[t].from < n) {
                pthread_create(&threads[t], NULL, bh_repulsion_worker, &tdata[t]);
            }
        }

        for (int t = 0; t < num_threads; t++) {
            if (tdata[t].from < n) {
                pthread_join(threads[t], NULL);
            }
        }
    }

    free(tree.nodes);
}

/* --- Gravity --- */

void c_gravity(
    const double* pos,
    const double* mass,
    double* forces,
    int n,
    double gravity,
    double scaling_ratio,
    int strong_gravity
) {
    for (int i = 0; i < n; i++) {
        double px = pos[2 * i];
        double py = pos[2 * i + 1];
        double dist = sqrt(px * px + py * py);
        if (dist > 0) {
            double factor;
            if (strong_gravity) {
                factor = scaling_ratio * mass[i] * (gravity / scaling_ratio);
            } else {
                factor = scaling_ratio * mass[i] * (gravity / scaling_ratio) / dist;
            }
            forces[2 * i] -= px * factor;
            forces[2 * i + 1] -= py * factor;
        }
    }
}

/* --- Attraction --- */

void c_attraction(
    const double* pos,
    const double* mass,
    const double* sizes,
    const int* edges_src,
    const int* edges_dst,
    const double* edges_weight,
    double* forces,
    int num_edges,
    double outbound_att_compensation,
    int lin_log_mode,
    int outbound_attraction_distribution,
    int adjust_sizes
) {
    for (int e = 0; e < num_edges; e++) {
        int src = edges_src[e];
        int dst = edges_dst[e];
        double w = edges_weight[e];

        double x_dist = pos[2 * src] - pos[2 * dst];
        double y_dist = pos[2 * src + 1] - pos[2 * dst + 1];
        double dist = sqrt(x_dist * x_dist + y_dist * y_dist);

        if (adjust_sizes && sizes) {
            dist -= (sizes[src] + sizes[dst]);
            if (dist <= 0) continue;

            double factor;
            if (lin_log_mode) {
                if (outbound_attraction_distribution) {
                    factor = -outbound_att_compensation * w * log1p(dist) / dist / mass[src];
                } else {
                    factor = -outbound_att_compensation * w * log1p(dist) / dist;
                }
            } else {
                if (outbound_attraction_distribution) {
                    factor = -outbound_att_compensation * w / mass[src];
                } else {
                    factor = -outbound_att_compensation * w;
                }
            }
            forces[2 * src] += x_dist * factor;
            forces[2 * src + 1] += y_dist * factor;
            forces[2 * dst] -= x_dist * factor;
            forces[2 * dst + 1] -= y_dist * factor;
        } else {
            if (lin_log_mode) {
                if (dist <= 0) continue;
                double factor;
                if (outbound_attraction_distribution) {
                    factor = -outbound_att_compensation * w * log1p(dist) / dist / mass[src];
                } else {
                    factor = -outbound_att_compensation * w * log1p(dist) / dist;
                }
                forces[2 * src] += x_dist * factor;
                forces[2 * src + 1] += y_dist * factor;
                forces[2 * dst] -= x_dist * factor;
                forces[2 * dst + 1] -= y_dist * factor;
            } else {
                double factor;
                if (outbound_attraction_distribution) {
                    factor = -outbound_att_compensation * w / mass[src];
                } else {
                    factor = -outbound_att_compensation * w;
                }
                forces[2 * src] += x_dist * factor;
                forces[2 * src + 1] += y_dist * factor;
                forces[2 * dst] -= x_dist * factor;
                forces[2 * dst + 1] -= y_dist * factor;
            }
        }
    }
}

/* --- Speed Adjustment & Position Update --- */

void c_step(
    double* pos,
    const double* mass,
    const double* sizes,
    const double* old_forces,
    const double* forces,
    int n,
    double* speed_ptr,
    double* speed_efficiency_ptr,
    double jitter_tolerance,
    int adjust_sizes
) {
    double speed = *speed_ptr;
    double speed_eff = *speed_efficiency_ptr;

    double total_swinging = 0.0;
    double total_effective_traction = 0.0;

    for (int i = 0; i < n; i++) {
        double d_dx = old_forces[2 * i] - forces[2 * i];
        double d_dy = old_forces[2 * i + 1] - forces[2 * i + 1];
        double swinging = sqrt(d_dx * d_dx + d_dy * d_dy);
        total_swinging += mass[i] * swinging;

        double s_dx = old_forces[2 * i] + forces[2 * i];
        double s_dy = old_forces[2 * i + 1] + forces[2 * i + 1];
        total_effective_traction += 0.5 * mass[i] * sqrt(s_dx * s_dx + s_dy * s_dy);
    }

    if (total_swinging == 0.0 || total_effective_traction == 0.0) {
        return;
    }

    double opt_jt = 0.05 * sqrt((double)n);
    double min_jt = sqrt(opt_jt);
    double max_jt = 10.0;
    double jt_val = (opt_jt * total_effective_traction) / ((double)n * (double)n);
    double clamped_jt = MAX(min_jt, MIN(max_jt, jt_val));
    double jt = jitter_tolerance * clamped_jt;

    double min_speed_eff = 0.05;
    if ((total_swinging / total_effective_traction) > 2.0) {
        if (speed_eff > min_speed_eff) {
            speed_eff *= 0.5;
        }
        jt = MAX(jt, jitter_tolerance);
    }

    double target_speed = (jt * speed_eff * total_effective_traction) / total_swinging;

    if (total_swinging > (jt * total_effective_traction)) {
        if (speed_eff > min_speed_eff) {
            speed_eff *= 0.7;
        }
    } else if (speed < 1000.0) {
        speed_eff *= 1.3;
    }

    double max_rise = 0.5;
    speed += MIN(target_speed - speed, max_rise * speed);

    *speed_ptr = speed;
    *speed_efficiency_ptr = speed_eff;

    for (int i = 0; i < n; i++) {
        double d_dx = old_forces[2 * i] - forces[2 * i];
        double d_dy = old_forces[2 * i + 1] - forces[2 * i + 1];
        double swinging = sqrt(d_dx * d_dx + d_dy * d_dy);
        double node_swinging = mass[i] * swinging;

        if (adjust_sizes && sizes) {
            double factor = (0.1 * speed) / (1.0 + sqrt(speed * node_swinging));
            double df = sqrt(forces[2 * i] * forces[2 * i] + forces[2 * i + 1] * forces[2 * i + 1]);
            if (df > 0.0) {
                factor = MIN(factor * df, 10.0) / df;
                pos[2 * i] += forces[2 * i] * factor;
                pos[2 * i + 1] += forces[2 * i + 1] * factor;
            }
        } else {
            double factor = speed / (1.0 + sqrt(speed * node_swinging));
            pos[2 * i] += forces[2 * i] * factor;
            pos[2 * i + 1] += forces[2 * i + 1] * factor;
        }
    }
}
