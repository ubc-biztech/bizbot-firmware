// Host test: the C runtime must reproduce the commands the sim sent.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bizbot_policy.h"
#include "test_vectors.h"

int main(void) {
    double worst_u = 0.0;
    int cmd_mismatch = 0, total = 0;
    for (int r = 0; r < TV_ROBOTS; ++r) {
        BizbotPolicyState s;
        bizbot_policy_reset(&s);
        for (int k = 0; k < TV_TICKS; ++k) {
            const float* x = &TV_SENSORS[r][k * 4];
            int16_t cmd = bizbot_policy_tick(&s, x[0], x[1], x[2], x[3]);
            double du = fabs((double)s.u - (double)TV_U[r][k]);
            if (du > worst_u) worst_u = du;
            int16_t want = (int16_t)(TV_U[r][k] * 500.0f);
            if (cmd != want && abs(cmd - want) > 1) ++cmd_mismatch;
            ++total;
        }
    }
    printf("ticks %d, worst |u_c - u_sim| = %.2e, commands off by more than 1 unit: %d\n",
           total, worst_u, cmd_mismatch);
    return (worst_u < 1e-3 && cmd_mismatch == 0) ? 0 : 1;
}
