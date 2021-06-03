         module tiling_interface
         use :: ISO_C_BINDING
         interface
           subroutine run_chameleon_dgemm_c(A, B, C, m, n, k,lda,&
             ldb, ldc, alpha, beta) bind(C)
           use :: ISO_C_BINDING
           implicit none
           integer(KIND=C_INT), value :: n,m,k
           integer(KIND=C_INT), value :: lda, ldb, ldc
           real(KIND=C_DOUBLE), value :: alpha, beta
           real(KIND=C_DOUBLE):: A(m,k)
           real(KIND=C_DOUBLE):: B(k,n)
           real(KIND=C_DOUBLE):: C(m,n)
           end subroutine run_chameleon_dgemm_c
           subroutine run_chameleon_dgemm_tile_c(A, B, C, m, n, k,lda,&
             ldb, ldc, alpha, beta) bind(C)
           use :: ISO_C_BINDING
           implicit none
           integer(KIND=C_INT), value :: n,m,k
           integer(KIND=C_INT), value :: lda, ldb, ldc
           real(KIND=C_DOUBLE), value :: alpha, beta
           real(KIND=C_DOUBLE):: A(m,k)
           real(KIND=C_DOUBLE):: B(k,n)
           real(KIND=C_DOUBLE):: C(m,n)
           end subroutine run_chameleon_dgemm_tile_c
           subroutine run_init_chameleon_c() bind(C)
           use :: ISO_C_BINDING
           implicit none
           end subroutine run_init_chameleon_c
           subroutine run_stop_chameleon_c() bind(C)
           use :: ISO_C_BINDING
           implicit none
           end subroutine run_stop_chameleon_c
           subroutine run_magma_dgemm_async_gpu_c(A, B, C, m, n, k,lda,&
             ldb, ldc) bind(C)
           use :: ISO_C_BINDING
           implicit none
           integer(KIND=C_INT), value :: n,m,k
           integer(KIND=C_INT), value :: lda, ldb, ldc
           real(KIND=C_DOUBLE):: A(m,k)
           real(KIND=C_DOUBLE):: B(k,n)
           real(KIND=C_DOUBLE):: C(m,n)
           end subroutine run_magma_dgemm_async_gpu_c
           subroutine run_starpu_dgemm_hybrid_c(A, B, C, m, n, k,&
               nslicesx, nslicesy, nsteps) bind(C)
           use :: ISO_C_BINDING
           implicit none
           integer(KIND=C_INT), value :: n,m,k
           integer(KIND=C_INT), value :: nslicesx, nslicesy, nsteps
           real(KIND=C_DOUBLE):: A(m,k)
           real(KIND=C_DOUBLE):: B(k,n)
           real(KIND=C_DOUBLE):: C(m,n)
           end subroutine run_starpu_dgemm_hybrid_c
           subroutine run_init_starpu_c() bind(C)
           use :: ISO_C_BINDING
           implicit none
           end subroutine run_init_starpu_c
           subroutine run_stop_starpu_c() bind(C)
           use :: ISO_C_BINDING
           implicit none
           end subroutine run_stop_starpu_c
         end interface
       end module

