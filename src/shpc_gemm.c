#include "assignment3.h"
#include<immintrin.h>
#include "omp.h"


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

        p++; 

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

        p++; 

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

        p++;
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


double *pack_A(double *A, int mc, int kc, int rsA, int csA, double *buff)
{
    int buff_index = 0;
    int count = 0;
    for (int ir = 0; ir < mc; ir += MR)
    {

        // general case
        if(mc - ir >= MR)
        {
            for (int p = 0; p < kc; p++)
            {
                for (int i = 0; i < MR; i++)
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
                for (int j = ib; j < MR; j++)
                {
                    buff_index++;
                }
            }
        }
    }
    return buff;
}

double *pack_B(double *B, int kc, int nc, int rsB, int csB, double *buff)
{
    int buff_index = 0;
    for (int jr = 0; jr < nc; jr += NR)
    {
        // general case. section is NR by kc
        if (NR <= nc - jr) {
            for (int p = 0; p < kc; p++)
            {
                for (int j = 0; j < NR; j++)
                {
                    buff[buff_index++] = *(B + csB * (jr + j) + rsB * (p));
                }
            }
        } else {

            // special case. section is <NR by kc
            int jb = nc - jr;
            for (int p = 0; p < kc; p++)
            {
                for (int j = 0; j < jb; j++)
                {
                    buff[buff_index++] = *(B + csB * (jr + j) + rsB * (p));
                }

                for (int j = jb; j < NR; j++)
                {
                    buff_index++;
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
    for (int jc = 0; jc < n; jc += NC)
    {
        int curr_NC = bli_min(NC, n - jc);

        for (int pc = 0; pc < k; pc += KC)
        {
            int curr_KC = bli_min(KC, k - pc);

            // pack B
            double *B_buff = (double *)_mm_malloc(KC * NC * sizeof(double), 64);
            double *B_panel = B + csB * jc + rsB * pc;
            pack_B(B_panel, curr_KC, curr_NC, rsB, csB, B_buff);

            
            for (int ic = 0; ic < m; ic += MC)
            {
                int curr_MC = bli_min(MC, m - ic);

                // pack A
                double *A_buff = (double *)_mm_malloc(KC * MC * sizeof(double), 64);
                double *A_panel = A + csA * pc + rsA * ic;
                pack_A(A_panel, curr_MC, curr_KC, rsA, csA, A_buff);

                #pragma omp parallel for
                for (int jr = 0; jr < curr_NC; jr += NR)
                {
                    int curr_NR = bli_min(NR, curr_NC - jr);

                    for (int ir = 0; ir < curr_MC; ir += MR)
                    {

                        int curr_MR = bli_min(MR, curr_MC - ir);
                        double *curr_C = C + rsC * (ir + ic) + csC * (jr + jc);
                        double *curr_A = A_buff + curr_KC * ir;
                        double *curr_B = B_buff + curr_KC * jr;

                        // general case
                        if ((curr_NR == NR && curr_MR == MR) && (rsC == 1)) 
                        {
                            ukernel(curr_KC, curr_A, 1, MR, curr_B, NR, 1, curr_C, rsC, csC);
                        }
                        else
                        {
                            double *edge_C = malloc((sizeof(double) * NR * MR));

                            for (int i = 0; i < curr_MR; i++){
                                for (int j = 0; j < curr_NR; j++){
                                    edge_C[i + j * MR] = curr_C[i * rsC + j * csC];
                                }
                            }

                            ukernel(curr_KC, curr_A, 1, MR, curr_B, NR, 1, edge_C, 1, MR);

                            for (int i = 0; i < curr_MR; i++){
                                for (int j = 0; j < curr_NR; j++){
                                    curr_C[i * rsC + j * csC] = edge_C[i + j * MR];
                                }
                            }
                            free(edge_C);
                        }
                    }
                }
                _mm_free(A_buff);
            }
            _mm_free(B_buff);
        }
   }
}
