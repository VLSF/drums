#include <math.h>
#include <stddef.h>

#define NODES 14

/* Polynomial interpolation of the tabulated Bessel function and its derivative. */
void interpolate(
    const double *table, int width, int rows,
    const double *x, int points, double step, double *values, double *derivatives
) {
    for (int j = 0; j < points; ++j) {
        double scaled = x[j] / step;
        int base = (int)floor(scaled) - NODES / 2 + 1;
        if (base < 0) base = 0;
        if (base > width - NODES) base = width - NODES;
        double local = scaled - base;
        double front[NODES + 1], back[NODES + 1];
        double front_derivative[NODES + 1], back_derivative[NODES + 1];
        double weight[NODES], derivative_weight[NODES];
        front[0] = back[NODES] = 1.;
        front_derivative[0] = back_derivative[NODES] = 0.;
        for (int i = 0; i < NODES; ++i) {
            int reflected = NODES - i - 1;
            front[i + 1] = front[i] * (local - i);
            front_derivative[i + 1] = front_derivative[i] * (local - i) + front[i];
            back[reflected] = back[reflected + 1] * (local - reflected);
            back_derivative[reflected] =
                back_derivative[reflected + 1] * (local - reflected) + back[reflected + 1];
        }
        double divisor = -6227020800.;  /* (-1)^13 13! */
        for (int i = 0; i < NODES; ++i) {
            weight[i] = front[i] * back[i + 1] / divisor;
            derivative_weight[i] =
                (front_derivative[i] * back[i + 1] + front[i] * back_derivative[i + 1]) / (divisor * step);
            if (i < NODES - 1) {
                divisor *= -(i + 1.) / (NODES - i - 1.);
            }
        }
        for (int i = 0; i < rows; ++i) {
            const double *sample = table + (size_t)i * width + base;
            double value = 0., derivative = 0.;
            for (int t = 0; t < NODES; ++t) {
                value += sample[t] * weight[t];
                derivative += sample[t] * derivative_weight[t];
            }
            values[(size_t)i * points + j] = value;
            derivatives[(size_t)i * points + j] = derivative;
        }
    }
}
