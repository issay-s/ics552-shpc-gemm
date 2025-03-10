#include "assignment3.h"
#include<immintrin.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
int mr = 4;
int nr = 4;
int kc = 48;
int mc = 48;
int nc = 48;


void shpc_dgemm(int m, int n, int k,
               double *A, int rsA, int csA,
               double *B, int rsB, int csB,
               double *C, int rsC, int csC)
{
  


   int jc, ic, pc, ir, jr;
   for (jc = 0; jc < n; jc += nc)
   {
      
       for (pc = 0; pc < k; pc += kc)
       {
          
           for (ic = 0; ic < m; ic += mc)
           {
              
               // double *A_buff = (double *) (malloc(sizeof(double) * mc * kc); // how would csA affect this?
               // double *B_buff = (double *) (malloc(sizeof(double) * kc * nr);




               for (jr = 0; jr < nc; jr += nr)
               {
                  
                   for (ir = 0; ir < mc; ir += mr)
                   {
                      
                       void *curr_A = A + csA * pc + rsA * (ic + ir);
                       void *curr_B = B + csB * (jc + jr) + rsB * pc;
                       void *curr_C = C + rsC * (ir + ic) + csC * (jr + jc);
                      
                       ukernel(kc, curr_A, rsA, csA, curr_B, rsB, csB, curr_C, rsC, csC);
                      
                   }
               }
           }
       }
   }
}


double *pack_A(double *A, int mc, int mr, int kc, int pc,int ic, int ir, int rsA, int csA, double *buff){
   int buff_index = 0;
   for(int ir = 0; ir < mc; ir += mr){
       for(int p = 0; p < kc; p++){
           memcpy(buff + buff_index, A + csA * (pc + p) + rsA * (ic + ir), sizeof(double));
           buff_index++;
       }
   }
   return buff;
}


double *pack_B(double *B, int jr, int kc, int nr, int pc, int rsB, int csB, double *buff){
   int buff_index = 0;
   for(int p = 0; p < kc; p++){
       for(int i = 0; i < nr; i++){
           memcpy(buff + buff_index, B + csB * (jr + i) + rsB * (pc + p), sizeof(double));
           buff_index++;
       }
   }
   return buff;
}




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


   for ( int p=0; p<k; p++ )
   {
       /* Declare vector register for load/broadcasting beta( p,j ) */
       __m256d beta_p_j;


       /* Declare a vector register to hold the current column of A and load
           it with the four elements of that column. */
       __m256d alpha_0123_0 = _mm256_loadu_pd(&A[0 * rsA + p * csA]);


       /* Load/broadcast beta( p,0 ). */
       beta_p_j = _mm256_broadcast_sd(&B[p * rsB + 0 * csB]);


       /* update the first column of C with the current column of A times
           beta ( p,0 ) */
       gamma_0123_0 = _mm256_fmadd_pd(alpha_0123_0, beta_p_j, gamma_0123_0);




       // second iteration of j loop
       /* Declare a vector register to hold the current column of A and load
           it with the four elements of that column. */
       __m256d alpha_0123_1 = _mm256_loadu_pd(&A[0 * rsA + p * csA]);


       /* Load/broadcast beta( p,1 ). */
       beta_p_j = _mm256_broadcast_sd(&B[p * rsB + 1 * csB]);


       /* update the second column of C with the current column of A times
           beta ( p,1 ) */
       gamma_0123_1 = _mm256_fmadd_pd(alpha_0123_1, beta_p_j, gamma_0123_1);




      
       // third iteration of j loop
       /* Declare a vector register to hold the current column of A and load
           it with the four elements of that column. */
       __m256d alpha_0123_2 = _mm256_loadu_pd(&A[0 * rsA + p * csA]);


       /* Load/broadcast beta( p,2 ). */
       beta_p_j = _mm256_broadcast_sd(&B[p * rsB + 2 * csB]);


       /* update the third column of C with the current column of A times
           beta ( p,2 ) */
       gamma_0123_2 = _mm256_fmadd_pd(alpha_0123_2, beta_p_j, gamma_0123_2);




       // fourth iteration of j loop
       /* Declare a vector register to hold the current column of A and load
           it with the four elements of that column. */
       __m256d alpha_0123_3 = _mm256_loadu_pd(&A[0 * rsA + p * csA]);


       /* Load/broadcast beta( p,3 ). */
       beta_p_j = _mm256_broadcast_sd(&B[p * rsB + 3 * csB]);


       /* update the fourth column of C with the current column of A times
           beta ( p,3 ) */
       gamma_0123_3 = _mm256_fmadd_pd(alpha_0123_3, beta_p_j, gamma_0123_3);




   }


   /* Store the updated results */
   _mm256_storeu_pd( &C[0*rsC + 0*csC], gamma_0123_0 );
   _mm256_storeu_pd( &C[0*rsC + 1*csC], gamma_0123_1 );
   _mm256_storeu_pd( &C[0*rsC + 2*csC], gamma_0123_2 );
   _mm256_storeu_pd( &C[0*rsC + 3*csC], gamma_0123_3 );   


   // _mm256_storeu_pd( &C[4*rsC + 0*csC], gamma_4567_0 );
   // _mm256_storeu_pd( &C[4*rsC + 1*csC], gamma_4567_1 );
   // _mm256_storeu_pd( &C[4*rsC + 2*csC], gamma_4567_2 );
   // _mm256_storeu_pd( &C[4*rsC + 3*csC], gamma_4567_3 );  
   // change Mr to 8?
}
