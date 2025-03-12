#include "assignment3.h"
#include<immintrin.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
int mr = 8;
int nr = 6;
int kc = 1571;
int mc = 152;
int nc = 4096;


void ukernel(
      int k,
      double *A, int rsA, int csA,
      double *B, int rsB, int csB,
      double *C, int rsC, int csC){




  /* Declare vector registers to hold 4x4 C and load them */
  __m256d gamma_0123_0 = _mm256_loadu_pd( &C[0*rsC + 0*csC] );
  __m256d gamma_0123_1 = _mm256_loadu_pd( &C[0*rsC + 1*csC] );
  __m256d gamma_0123_2 = _mm256_loadu_pd( &C[0*rsC + 2*csC] );
  __m256d gamma_0123_3 = _mm256_loadu_pd( &C[0*rsC + 3*csC] );
  __m256d gamma_0123_4 = _mm256_loadu_pd(&C[0 * rsC + 4 * csC]);
  __m256d gamma_0123_5 = _mm256_loadu_pd(&C[0 * rsC + 5 * csC]);

  __m256d gamma_4567_0 = _mm256_loadu_pd( &C[4*rsC + 0*csC] );
  __m256d gamma_4567_1 = _mm256_loadu_pd( &C[4*rsC + 1*csC] );
  __m256d gamma_4567_2 = _mm256_loadu_pd( &C[4*rsC + 2*csC] );
  __m256d gamma_4567_3 = _mm256_loadu_pd( &C[4*rsC + 3*csC] );
  __m256d gamma_4567_4 = _mm256_loadu_pd(&C[4 * rsC + 4 * csC]);
  __m256d gamma_4567_5 = _mm256_loadu_pd(&C[4 * rsC + 5 * csC]);

      /* Declare vector register for load/broadcasting beta( p,j ) */
__m256d beta_p_j;


  for ( int p=0; p<k; p++ )
  {


      /* Declare a vector register to hold the current column of A and load
          it with the four elements of that column. */
      __m256d alpha_0123_p = _mm256_loadu_pd(&A[0 * rsA + p * csA]);
      __m256d alpha_4567_p = _mm256_loadu_pd(&A[4 * rsA + p * csA]);

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


  }


  /* Store the updated results */
  _mm256_storeu_pd( &C[0*rsC + 0*csC], gamma_0123_0 );
  _mm256_storeu_pd( &C[0*rsC + 1*csC], gamma_0123_1 );
  _mm256_storeu_pd( &C[0*rsC + 2*csC], gamma_0123_2 );
  _mm256_storeu_pd( &C[0*rsC + 3*csC], gamma_0123_3 );  
  _mm256_storeu_pd( &C[0*rsC + 4*csC], gamma_0123_4 );
  _mm256_storeu_pd( &C[0*rsC + 5*csC], gamma_0123_5 );  

  _mm256_storeu_pd( &C[4*rsC + 0*csC], gamma_4567_0 );
  _mm256_storeu_pd( &C[4*rsC + 1*csC], gamma_4567_1 );
  _mm256_storeu_pd( &C[4*rsC + 2*csC], gamma_4567_2 );
  _mm256_storeu_pd( &C[4*rsC + 3*csC], gamma_4567_3 ); 
  _mm256_storeu_pd( &C[4*rsC + 4*csC], gamma_4567_4 );
  _mm256_storeu_pd( &C[4*rsC + 5*csC], gamma_4567_5 ); 
}




double *pack_A(double *A, int mc, int mr, int kc, int rsA, int csA, double *buff){
  int buff_index = 0;
  for(int ir = 0; ir < mc; ir += mr){
      for(int p = 0; p < kc; p++){
          for(int i = 0; i < mr; i++){
               buff[buff_index++] = *(A + csA * (p) + rsA * (ir + i));
          }
      }
  }
  return buff;
}




double *pack_B(double *B, int kc, int nc, int nr, int rsB, int csB, double *buff){
  int buff_index = 0;
  for(int j = 0; j < nc; j += nr){
      for (int p = 0; p < kc; p++)
      {
          for (int i = 0; i < nr; i++)
          {
               buff[buff_index++] = *(B + csB * (j + i) + rsB * (p));
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


   int jc, ic, pc, ir, jr;
   for (jc = 0; jc < n; jc += nc)
   {
    nc = bli_min(nc, n - jc);

       for (pc = 0; pc < k; pc += kc)
       {
        kc = bli_min(kc, k - pc);
           // pack B
           double *B_buff = (double *)(malloc(sizeof(double) * kc * nc));
           double *B_panel = B + csB * jc + rsB * pc;
           pack_B(B_panel, kc, nc, nr, rsB, csB, B_buff);


           for (ic = 0; ic < m; ic += mc)
           {
              mc = bli_min(mc, m - ic);
               // pack A
               double *A_buff = (double *)(malloc(sizeof(double) * mc * kc));
               double *A_panel = A + csA * pc + rsA * ic;
               pack_A(A_panel, mc, mr, kc, rsA, csA, A_buff);

               for (jr = 0; jr < nc; jr += nr)
               {
                   for (ir = 0; ir < mc; ir += mr)
                   {
                       double *curr_C = C + rsC * (ir + ic) + csC * (jr + jc);
                       double *curr_A = A_buff + kc * ir;
                       double *curr_B = B_buff + kc * jr;

                       ukernel(kc, curr_A, 1, mr, curr_B, nr, 1, curr_C, rsC, csC);
                   }
               }
               free(A_buff);
           }
           free(B_buff);
       }
   }
}

// TODO free everything
