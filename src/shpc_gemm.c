#include "assignment3.h"
#include<immintrin.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "omp.h"
int mr = 8;
int nr = 6;
int kc = 256;
int mc = 960;
// int mc = 72;
int nc = 11256;
int UNROLLING = 2;
// TODO move to different file

void ukernel(
    int k,
    double *A, int rsA, int csA,
    double *B, int rsB, int csB,
    double *C, int rsC, int csC)
{

    /* Declare vector registers to hold 4x4 C and load them */
    __m256d gamma_0123_0 = _mm256_loadu_pd(&C[0 * rsC + 0 * csC]);
    __m256d gamma_0123_1 = _mm256_loadu_pd(&C[0 * rsC + 1 * csC]);
    __m256d gamma_0123_2 = _mm256_loadu_pd(&C[0 * rsC + 2 * csC]);
    __m256d gamma_0123_3 = _mm256_loadu_pd(&C[0 * rsC + 3 * csC]);
    __m256d gamma_0123_4 = _mm256_loadu_pd(&C[0 * rsC + 4 * csC]);
    __m256d gamma_0123_5 = _mm256_loadu_pd(&C[0 * rsC + 5 * csC]);

    __m256d gamma_4567_0 = _mm256_loadu_pd(&C[4 * rsC + 0 * csC]);
    __m256d gamma_4567_1 = _mm256_loadu_pd(&C[4 * rsC + 1 * csC]);
    __m256d gamma_4567_2 = _mm256_loadu_pd(&C[4 * rsC + 2 * csC]);
    __m256d gamma_4567_3 = _mm256_loadu_pd(&C[4 * rsC + 3 * csC]);
    __m256d gamma_4567_4 = _mm256_loadu_pd(&C[4 * rsC + 4 * csC]);
    __m256d gamma_4567_5 = _mm256_loadu_pd(&C[4 * rsC + 5 * csC]);

    /* Declare vector register for load/broadcasting beta( p,j ) */
    __m256d beta_p_j;
    __m256d alpha_0123_p;
    __m256d alpha_4567_p;

    int p = 0;
    while (p < k - UNROLLING)
    {

        /* Declare a vector register to hold the current column of A and load
            it with the four elements of that column. */
        alpha_0123_p = _mm256_loadu_pd(&A[0 * rsA + p * csA]);
        alpha_4567_p = _mm256_loadu_pd(&A[4 * rsA + p * csA]);

        /* Load/broadcast beta( p,0 ). */
        beta_p_j = _mm256_broadcast_sd(&B[p * rsB + 0 * csB]);

        /* update the first column of C with the current column of A times
            beta ( p,0 ) */
        gamma_0123_0 = _mm256_fmadd_pd(alpha_0123_p, beta_p_j, gamma_0123_0);
        gamma_4567_0 = _mm256_fmadd_pd(alpha_4567_p, beta_p_j, gamma_4567_0);

        // second iteration of j loop
        /* Load/broadcast beta( p,1 ). */
        beta_p_j = _mm256_broadcast_sd(&B[p * rsB + 1 * csB]);

        /* update the second column of C with the current column of A times
            beta ( p,1 ) */
        gamma_0123_1 = _mm256_fmadd_pd(alpha_0123_p, beta_p_j, gamma_0123_1);
        gamma_4567_1 = _mm256_fmadd_pd(alpha_4567_p, beta_p_j, gamma_4567_1);

        // third iteration of j loop

        /* Load/broadcast beta( p,2 ). */
        beta_p_j = _mm256_broadcast_sd(&B[p * rsB + 2 * csB]);

        /* update the third column of C with the current column of A times
            beta ( p,2 ) */
        gamma_0123_2 = _mm256_fmadd_pd(alpha_0123_p, beta_p_j, gamma_0123_2);
        gamma_4567_2 = _mm256_fmadd_pd(alpha_4567_p, beta_p_j, gamma_4567_2);

        // fourth iteration of j loop

        /* Load/broadcast beta( p,3 ). */
        beta_p_j = _mm256_broadcast_sd(&B[p * rsB + 3 * csB]);

        /* update the fourth column of C with the current column of A times
            beta ( p,3 ) */
        gamma_0123_3 = _mm256_fmadd_pd(alpha_0123_p, beta_p_j, gamma_0123_3);
        gamma_4567_3 = _mm256_fmadd_pd(alpha_4567_p, beta_p_j, gamma_4567_3);

        // fifth iteration of j loop

        /* Load/broadcast beta( p,2 ). */
        beta_p_j = _mm256_broadcast_sd(&B[p * rsB + 4 * csB]);

        /* update the third column of C with the current column of A times
            beta ( p,2 ) */
        gamma_0123_4 = _mm256_fmadd_pd(alpha_0123_p, beta_p_j, gamma_0123_4);
        gamma_4567_4 = _mm256_fmadd_pd(alpha_4567_p, beta_p_j, gamma_4567_4);

        // sixth iteration of j loop

        /* Load/broadcast beta( p,3 ). */
        beta_p_j = _mm256_broadcast_sd(&B[p * rsB + 5 * csB]);

        /* update the fourth column of C with the current column of A times
            beta ( p,3 ) */
        gamma_0123_5 = _mm256_fmadd_pd(alpha_0123_p, beta_p_j, gamma_0123_5);
        gamma_4567_5 = _mm256_fmadd_pd(alpha_4567_p, beta_p_j, gamma_4567_5);

        p++; // TODO this is an increment

                /* Declare a vector register to hold the current column of A and load
            it with the four elements of that column. */
        alpha_0123_p = _mm256_loadu_pd(&A[0 * rsA + p * csA]);
        alpha_4567_p = _mm256_loadu_pd(&A[4 * rsA + p * csA]);

        /* Load/broadcast beta( p,0 ). */
        beta_p_j = _mm256_broadcast_sd(&B[p * rsB + 0 * csB]);

        /* update the first column of C with the current column of A times
            beta ( p,0 ) */
        gamma_0123_0 = _mm256_fmadd_pd(alpha_0123_p, beta_p_j, gamma_0123_0);
        gamma_4567_0 = _mm256_fmadd_pd(alpha_4567_p, beta_p_j, gamma_4567_0);

        // second iteration of j loop
        /* Load/broadcast beta( p,1 ). */
        beta_p_j = _mm256_broadcast_sd(&B[p * rsB + 1 * csB]);

        /* update the second column of C with the current column of A times
            beta ( p,1 ) */
        gamma_0123_1 = _mm256_fmadd_pd(alpha_0123_p, beta_p_j, gamma_0123_1);
        gamma_4567_1 = _mm256_fmadd_pd(alpha_4567_p, beta_p_j, gamma_4567_1);

        // third iteration of j loop

        /* Load/broadcast beta( p,2 ). */
        beta_p_j = _mm256_broadcast_sd(&B[p * rsB + 2 * csB]);

        /* update the third column of C with the current column of A times
            beta ( p,2 ) */
        gamma_0123_2 = _mm256_fmadd_pd(alpha_0123_p, beta_p_j, gamma_0123_2);
        gamma_4567_2 = _mm256_fmadd_pd(alpha_4567_p, beta_p_j, gamma_4567_2);

        // fourth iteration of j loop

        /* Load/broadcast beta( p,3 ). */
        beta_p_j = _mm256_broadcast_sd(&B[p * rsB + 3 * csB]);

        /* update the fourth column of C with the current column of A times
            beta ( p,3 ) */
        gamma_0123_3 = _mm256_fmadd_pd(alpha_0123_p, beta_p_j, gamma_0123_3);
        gamma_4567_3 = _mm256_fmadd_pd(alpha_4567_p, beta_p_j, gamma_4567_3);

        // fifth iteration of j loop

        /* Load/broadcast beta( p,2 ). */
        beta_p_j = _mm256_broadcast_sd(&B[p * rsB + 4 * csB]);

        /* update the third column of C with the current column of A times
            beta ( p,2 ) */
        gamma_0123_4 = _mm256_fmadd_pd(alpha_0123_p, beta_p_j, gamma_0123_4);
        gamma_4567_4 = _mm256_fmadd_pd(alpha_4567_p, beta_p_j, gamma_4567_4);

        // sixth iteration of j loop

        /* Load/broadcast beta( p,3 ). */
        beta_p_j = _mm256_broadcast_sd(&B[p * rsB + 5 * csB]);

        /* update the fourth column of C with the current column of A times
            beta ( p,3 ) */
        gamma_0123_5 = _mm256_fmadd_pd(alpha_0123_p, beta_p_j, gamma_0123_5);
        gamma_4567_5 = _mm256_fmadd_pd(alpha_4567_p, beta_p_j, gamma_4567_5);

        p++; // TODO this is an increment

    }

    while(p < k)
    {
        /* Declare a vector register to hold the current column of A and load
            it with the four elements of that column. */
        alpha_0123_p = _mm256_loadu_pd(&A[0 * rsA + p * csA]);
        alpha_4567_p = _mm256_loadu_pd(&A[4 * rsA + p * csA]);

        /* Load/broadcast beta( p,0 ). */
        beta_p_j = _mm256_broadcast_sd(&B[p * rsB + 0 * csB]);

        /* update the first column of C with the current column of A times
            beta ( p,0 ) */
        gamma_0123_0 = _mm256_fmadd_pd(alpha_0123_p, beta_p_j, gamma_0123_0);
        gamma_4567_0 = _mm256_fmadd_pd(alpha_4567_p, beta_p_j, gamma_4567_0);

        // second iteration of j loop
        /* Load/broadcast beta( p,1 ). */
        beta_p_j = _mm256_broadcast_sd(&B[p * rsB + 1 * csB]);

        /* update the second column of C with the current column of A times
            beta ( p,1 ) */
        gamma_0123_1 = _mm256_fmadd_pd(alpha_0123_p, beta_p_j, gamma_0123_1);
        gamma_4567_1 = _mm256_fmadd_pd(alpha_4567_p, beta_p_j, gamma_4567_1);

        // third iteration of j loop

        /* Load/broadcast beta( p,2 ). */
        beta_p_j = _mm256_broadcast_sd(&B[p * rsB + 2 * csB]);

        /* update the third column of C with the current column of A times
            beta ( p,2 ) */
        gamma_0123_2 = _mm256_fmadd_pd(alpha_0123_p, beta_p_j, gamma_0123_2);
        gamma_4567_2 = _mm256_fmadd_pd(alpha_4567_p, beta_p_j, gamma_4567_2);

        // fourth iteration of j loop

        /* Load/broadcast beta( p,3 ). */
        beta_p_j = _mm256_broadcast_sd(&B[p * rsB + 3 * csB]);

        /* update the fourth column of C with the current column of A times
            beta ( p,3 ) */
        gamma_0123_3 = _mm256_fmadd_pd(alpha_0123_p, beta_p_j, gamma_0123_3);
        gamma_4567_3 = _mm256_fmadd_pd(alpha_4567_p, beta_p_j, gamma_4567_3);

        // fifth iteration of j loop

        /* Load/broadcast beta( p,2 ). */
        beta_p_j = _mm256_broadcast_sd(&B[p * rsB + 4 * csB]);

        /* update the third column of C with the current column of A times
            beta ( p,2 ) */
        gamma_0123_4 = _mm256_fmadd_pd(alpha_0123_p, beta_p_j, gamma_0123_4);
        gamma_4567_4 = _mm256_fmadd_pd(alpha_4567_p, beta_p_j, gamma_4567_4);

        // sixth iteration of j loop

        /* Load/broadcast beta( p,3 ). */
        beta_p_j = _mm256_broadcast_sd(&B[p * rsB + 5 * csB]);

        /* update the fourth column of C with the current column of A times
            beta ( p,3 ) */
        gamma_0123_5 = _mm256_fmadd_pd(alpha_0123_p, beta_p_j, gamma_0123_5);
        gamma_4567_5 = _mm256_fmadd_pd(alpha_4567_p, beta_p_j, gamma_4567_5);

        p++; // TODO this is an increment
    }

    /* Store the updated results */
    _mm256_storeu_pd(&C[0 * rsC + 0 * csC], gamma_0123_0);
    _mm256_storeu_pd(&C[0 * rsC + 1 * csC], gamma_0123_1);
    _mm256_storeu_pd(&C[0 * rsC + 2 * csC], gamma_0123_2);
    _mm256_storeu_pd(&C[0 * rsC + 3 * csC], gamma_0123_3);
    _mm256_storeu_pd(&C[0 * rsC + 4 * csC], gamma_0123_4);
    _mm256_storeu_pd(&C[0 * rsC + 5 * csC], gamma_0123_5);

    _mm256_storeu_pd(&C[4 * rsC + 0 * csC], gamma_4567_0);
    _mm256_storeu_pd(&C[4 * rsC + 1 * csC], gamma_4567_1);
    _mm256_storeu_pd(&C[4 * rsC + 2 * csC], gamma_4567_2);
    _mm256_storeu_pd(&C[4 * rsC + 3 * csC], gamma_4567_3);
    _mm256_storeu_pd(&C[4 * rsC + 4 * csC], gamma_4567_4);
    _mm256_storeu_pd(&C[4 * rsC + 5 * csC], gamma_4567_5);
}


double *pack_A(double *A, int mc, int kc, int mod_kc, int rsA, int csA, double *buff)
{
    int buff_index = 0;
    int count = 0;
    for (int ir = 0; ir < mc; ir += mr)
    {
        // general case
        if(mc - ir >= mr)
        {
            for (int p = 0; p < kc; p++)
            {
                for (int i = 0; i < mr; i++)
                {
                    buff[buff_index++] = *(A + csA * (p) + rsA * (ir + i));
                }
            }
        }
        // special case 
        else {

            int ib = mc - ir;
            for (int p = 0; p < kc; p++)
            {
                for (int i = 0; i < ib; i++)
                {
                    buff[buff_index++] = *(A + csA * (p) + rsA * (ir + i));
                    count++;
                }
                for (int j = ib; j < mr; j++)
                {
                    // assert(buff_index < mc * kc);
                    buff[buff_index++] = 0.0;
                }
            }
        }
        
    }
    return buff;
}

double *pack_B(double *B, int kc, int mod_kc, int nc, int rsB, int csB, double *buff)
{
    int buff_index = 0;
    for (int j = 0; j < nc; j += nr)
    {
        // general case. section is nr by kc
        if (nr <= nc - j) {
            for (int p = 0; p < kc; p++)
            {
                for (int i = 0; i < nr; i++)
                {
                    buff[buff_index++] = *(B + csB * (j + i) + rsB * (p));
                }
            }
        } else {
            // special case. section is <nr by kc
            int jb = nc - j;
            for (int p = 0; p < kc; p++)
            {
                for (int i = 0; i < jb; i++)
                {
                    buff[buff_index++] = *(B + csB * (j + i) + rsB * (p));
                }
                for (int i = jb; i < nr; i++)
                {
                    buff[buff_index++] = 0.0;
                }
            }
        }

    }
    return buff;
}

void shpc_dgemm(int m, int n, int k,
              double *A, int rsA, int csA,
              double *B, int rsB, int csB,
              double *C, int rsC, int csC)
{
    double *A_buff = (double *)_mm_malloc(kc * mc * sizeof(double), 64);
    double *B_buff = (double *)_mm_malloc(kc * nc * sizeof(double), 64);
    int jc, ic, pc, ir, jr;
    int curr_nc, curr_kc, curr_mc, curr_nr, curr_mr;
    for (jc = 0; jc < n; jc += nc)
    {
        curr_nc = bli_min(nc, n - jc);

        for (pc = 0; pc < k; pc += kc)
        {
            curr_kc = bli_min(kc, k - pc);
            int mod_kc = curr_kc;
            while (mod_kc % UNROLLING != 0)
                mod_kc++;

            // pack B
            double *B_panel = B + csB * jc + rsB * pc;
            pack_B(B_panel, curr_kc, mod_kc, curr_nc, rsB, csB, B_buff);

            // #pragma omp parallel for
            for (ic = 0; ic < m; ic += mc)
            {
                curr_mc = bli_min(mc, m - ic);
                // pack A
                double *A_panel = A + csA * pc + rsA * ic;
                pack_A(A_panel, curr_mc, curr_kc, mod_kc, rsA, csA, A_buff);

                for (jr = 0; jr < curr_nc; jr += nr)
                {

                    curr_nr = bli_min(nr, curr_nc - jr);

                    for (ir = 0; ir < curr_mc; ir += mr)
                    {

                        curr_mr = bli_min(mr, curr_mc - ir);

                        double *curr_C = C + rsC * (ir + ic) + csC * (jr + jc);
                        double *curr_A = A_buff + curr_kc * ir;
                        double *curr_B = B_buff + curr_kc * jr;
                        if (curr_nr == nr && curr_mr == mr && csC == 1)
                        {
                            ukernel(curr_kc, curr_A, 1, mr, curr_B, nr, 1, curr_C, rsC, csC);
                        }
                        else
                        {
                            double *edge_C = malloc((sizeof(double) * nr * mr));

                            for (int i = 0; i < curr_mr; i++){
                                for (int j = 0; j < curr_nr; j++){
                                    edge_C[i + j * mr] = curr_C[i * rsC + j * csC];
                                }
                            }

                            ukernel(curr_kc, curr_A, 1, mr, curr_B, nr, 1, edge_C, 1, mr);

                            for (int i = 0; i < curr_mr; i++){
                                for (int j = 0; j < curr_nr; j++){
                                    curr_C[i * rsC + j * csC] = edge_C[i + j * mr];
                                }
                            }
                            free(edge_C);
                        }
                    }
                }
            }
        }
   }
   _mm_free(A_buff);
   _mm_free(B_buff);
}



//TODO csX issue is with the edge case
// something is wrong with padding kc (buffer index?)
// parallelize
// please