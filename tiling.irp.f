BEGIN_PROVIDER [ integer, tile_size ]
implicit none
BEGIN_DOC
! Tile size for tiling tables
END_DOC
tile_size = 24
END_PROVIDER

 BEGIN_PROVIDER [double precision, rescale_een_e_tiled, (tile_size, tile_size, 0:ntiles_nelec, 0:ntiles_nelec, 0:ncord)]
!&BEGIN_PROVIDER [double precision, rescale_een_e_tiled_T, (tile_size, tile_size, 0:ntiles_nelec, 0:ntiles_nelec, 0:ncord)]
&BEGIN_PROVIDER [double precision, rescale_een_n_tiled, (tile_size, tile_size, 0:ncord, 0:ntiles_nelec, 0:ntiles_nnuc)]
&BEGIN_PROVIDER [double precision, rescale_een_e_deriv_e_tiled, (tile_size, tile_size, 4, 0:ntiles_nelec, 0:ntiles_nelec, 0:ncord )]
!&BEGIN_PROVIDER [double precision, rescale_een_e_deriv_e_tiled_T, (tile_size, tile_size, 4, 0:ntiles_nelec, 0:ntiles_nelec, 0:ncord )]
 implicit none
 BEGIN_DOC
 ! R = exp(-kappa r) for electron-electron for $J_{een}$
 END_DOC
 integer :: i, j, k, l, a, ii, jj, aa
 integer :: idxi, idxj, idxa

 ! Fill up rescale_een_e_tiled
 do l = 0, ncord
    do j = 0, ntiles_nelec - 1
     do i = 0, ntiles_nelec - 1
       do jj = 1, tile_size
         idxj = j*tile_size + jj
         do ii = 1, tile_size
           idxi = i*tile_size + ii
           rescale_een_e_tiled(ii,jj,i,j,l) = rescale_een_e(idxi,idxj,l)
         enddo
       enddo
    enddo
    enddo
 enddo

 !! Fill up rescale_een_e_tiled_T
 !do l = 0, ncord
 !   do j = 0, ntiles_nelec - 1
 !    do i = 0, ntiles_nelec - 1
 !      do jj = 1, tile_size
 !        idxj = j*tile_size + jj
 !        do ii = 1, tile_size
 !          idxi = i*tile_size + ii
 !          rescale_een_e_tiled_T(jj,ii,i,j,l) = rescale_een_e(idxi,idxj,l)
 !        enddo
 !      enddo
 !   enddo
 !   enddo
 !enddo

 ! Fill up rescale_een_n_tiled
 do l = 0, ncord
    do a = 0, ntiles_nnuc - 1
       do i = 0, ntiles_nelec - 1
         do ii = 1, tile_size
           idxi = i*tile_size + ii
           do aa = 1, tile_size
             idxa = a*tile_size + aa
             rescale_een_n_tiled(ii, aa, l, i, a) =  rescale_een_n(idxi, idxa, l)
           enddo
         enddo
       enddo
    enddo
 enddo

 ! Fill up rescale_een_e_deriv_e_tiled
 do l = 0, ncord
    do j = 0, ntiles_nelec - 1
    do jj = 1, tile_size
       idxj = j*tile_size + jj
       do i = 0, ntiles_nelec - 1
       do ii = 1, tile_size
          idxi = i*tile_size + ii
          do k = 1, 4
             rescale_een_e_deriv_e_tiled(ii,jj,k,i,j,l) = rescale_een_e_deriv_e(idxi, k, idxj, l) 
          enddo
       enddo
       enddo
    enddo
    enddo
enddo

! ! Fill up rescale_een_e_deriv_e_tiled_T
! do l = 0, ncord
!    do j = 0, ntiles_nelec - 1
!    do jj = 1, tile_size
!       idxj = j*tile_size + jj
!       do i = 0, ntiles_nelec - 1
!       do ii = 1, tile_size
!          idxi = i*tile_size + ii
!          do k = 1, 4
!             rescale_een_e_deriv_e_tiled_T(jj,ii,k,i,j,l) = rescale_een_e_deriv_e(idxi, k, idxj, l) 
!          enddo
!       enddo
!       enddo
!    enddo
!    enddo
!enddo

END_PROVIDER

 BEGIN_PROVIDER [ double precision,  tmp_c_tiled_nounroll, (tile_size, tile_size,0:ncord, 0:ntiles_nelec, 0:ntiles_nnuc,0:ncord-1) ]
&BEGIN_PROVIDER [ double precision, dtmp_c_tiled_nounroll, (tile_size, tile_size,4,0:ncord, 0:ntiles_nelec, 0:ntiles_nnuc,0:ncord-1) ]
 !use tiling_interface
 implicit none
 BEGIN_DOC
 ! Calculate the intermediate buffers
 ! tmp_c:
 ! r_{ij}^k . R_{ja}^l -> tmp_c_{ia}^{kl}
 !
 ! dtmp_c:
 ! dr_{ij}^k . R_{ja}^l -> dtmp_c_{ia}^{kl}
 END_DOC
 integer :: k, i, j, a, l,m
 integer :: ii, jj, aa, kk, ll
 integer :: res

 ! r_{ij}^k . R_{ja}^l -> tmp_c_{ia}^{kl}
 !$OMP parallel                           &
 !$OMP private(m, jj, kk, ii, ll)         &
 !$OMP default (shared)                   
 !$OMP do collapse(3) schedule(dynamic)
 do k=0,ncord-1
   do j = 0, ntiles_nelec - 1
     do a = 0, ntiles_nnuc - 1
       do i = 0, ntiles_nelec - 1
        do m = 0, ncord
          do jj = 1, tile_size
            do kk = 1, tile_size
              do ii = 1, tile_size
                 tmp_c_tiled_nounroll(ii,jj,m,j,a,k) = tmp_c_tiled_nounroll(ii,jj,m,j,a,k) + &
                                          rescale_een_e_tiled(ii,kk+0,j,i,k)*&
                                          rescale_een_n_tiled(kk+0,jj,m,i,a)
            enddo
           enddo
         enddo
        enddo
       enddo
     enddo
   enddo
 enddo
 !$OMP end do

 ! dr_{ij}^k . R_{ja}^l -> dtmp_c_{ia}^{kl}
 !$OMP do collapse(3) schedule(dynamic)
 do k=0,ncord-1
   do a = 0, ntiles_nnuc - 1
     do j = 0, ntiles_nelec - 1
       do i = 0, ntiles_nelec - 1
        do m = 0, ncord
          do ll = 1, 4
           do jj = 1, tile_size
             do kk = 1, tile_size
               do ii = 1, tile_size
                 dtmp_c_tiled_nounroll(ii,jj,ll,m,j,a,k) =       dtmp_c_tiled_nounroll(ii,jj,ll,m,j,a,k)  + &
                                       rescale_een_e_deriv_e_tiled(ii,kk+0,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+0,jj,m,i,a)  
             enddo
            enddo
           enddo
         enddo
        enddo
       enddo
     enddo
   enddo
 enddo
 !$OMP end do
 !$OMP end parallel 


END_PROVIDER


 BEGIN_PROVIDER [ double precision,  tmp_c_tiled, (tile_size, tile_size,0:ncord, 0:ntiles_nelec, 0:ntiles_nnuc,0:ncord-1) ]
&BEGIN_PROVIDER [ double precision, dtmp_c_tiled, (tile_size, tile_size,4,0:ncord, 0:ntiles_nelec, 0:ntiles_nnuc,0:ncord-1) ]
 !use tiling_interface
 implicit none
 BEGIN_DOC
 ! Calculate the intermediate buffers
 ! tmp_c:
 ! r_{ij}^k . R_{ja}^l -> tmp_c_{ia}^{kl}
 !
 ! dtmp_c:
 ! dr_{ij}^k . R_{ja}^l -> dtmp_c_{ia}^{kl}
 END_DOC
 integer :: k, i, j, a, l,m
 integer :: ii, jj, aa, kk, ll
 integer :: res

 ! r_{ij}^k . R_{ja}^l -> tmp_c_{ia}^{kl}
 !$OMP parallel                           &
 !$OMP private(m, jj, kk, ii, ll)         &
 !$OMP default (shared)                   
 !$OMP do collapse(3) schedule(dynamic)
 do k=0,ncord-1
   do j = 0, ntiles_nelec - 1
     do a = 0, ntiles_nnuc - 1
       do i = 0, ntiles_nelec - 1
   call dgemm('N','N', tile_size, tile_size*(ncord+1), tile_size, 1.d0,           &
       rescale_een_e_tiled(1,1,j,i,k), size(rescale_een_e_tiled,1),                  &
       rescale_een_n_tiled(1,1,0,i,a), size(rescale_een_n_tiled,1), 1.d0,            &
       tmp_c_tiled(1,1,0,j,a,k), size(tmp_c_tiled,1))
   !call run_magma_dgemm_async_gpu_c(rescale_een_e_tiled(1,1,j,i,k),       &
   !                                rescale_een_n_tiled(1,1,0,i,a), &
   !                                tmp_c_tiled(1,1,0,j,a,k),       &
   !                                tile_size, tile_size*(ncord+1), &
   !                                tile_size,                      &
   !                                size(rescale_een_e_tiled,1),    &
   !                                size(rescale_een_n_tiled,1),    &
   !                                size(tmp_c_tiled,1))
   !     do m = 0, ncord
   !           !DIR$ vector aligned
   !       do jj = 1, tile_size
   !         !DIR$ vector aligned
   !         do kk = 1, tile_size, 4
   !           !DIR$ vector aligned
   !           do ii = 1, tile_size
   !              tmp_c_tiled(ii,jj,m,j,a,k) = tmp_c_tiled(ii,jj,m,j,a,k) + &
   !                                       rescale_een_e_tiled(ii,kk+0,j,i,k)*&
   !                                       rescale_een_n_tiled(kk+0,jj,m,i,a)+&
   !                                       rescale_een_e_tiled(ii,kk+1,j,i,k)*&
   !                                       rescale_een_n_tiled(kk+1,jj,m,i,a)+&
   !                                       rescale_een_e_tiled(ii,kk+2,j,i,k)*&
   !                                       rescale_een_n_tiled(kk+2,jj,m,i,a)+&
   !                                       rescale_een_e_tiled(ii,kk+3,j,i,k)*&
   !                                       rescale_een_n_tiled(kk+3,jj,m,i,a)
   !         enddo
   !        enddo
   !      enddo
   !     enddo
       enddo
     enddo
   enddo
 enddo
 !$OMP end do

 ! dr_{ij}^k . R_{ja}^l -> dtmp_c_{ia}^{kl}
 !$OMP do collapse(3) schedule(dynamic)
 do k=0,ncord-1
   do a = 0, ntiles_nnuc - 1
     do j = 0, ntiles_nelec - 1
       do i = 0, ntiles_nelec - 1
   call dgemm('N','N', 4*tile_size, tile_size*(ncord+1), tile_size, 1.d0,         &
       rescale_een_e_deriv_e_tiled(1,1,1,j,i,k), 4*size(rescale_een_e_deriv_e_tiled,1),&
       rescale_een_n_tiled(1,1,0,i,a), size(rescale_een_n_tiled,1), 1.d0,            &
       dtmp_c_tiled(1,1,1,0,j,a,k), 4*size(dtmp_c_tiled,1))

   !call run_magma_dgemm_async_gpu_c(rescale_een_e_deriv_e_tiled(1,1,1,k,j,i), &
   !                                rescale_een_n_tiled(1,1,0,i,a),            &
   !                                dtmp_c_tiled(1,1,1,0,j,a,k),               &
   !                                4*tile_size, tile_size*(ncord+1),          &
   !                                tile_size,                                 &
   !                                4*size(rescale_een_e_deriv_e_tiled,1),     &
   !                                size(rescale_een_n_tiled,1),               &
   !                                4*size(dtmp_c_tiled,1))
   !    do m = 0, ncord
   !          !DIR$ vector aligned
   !      do ll = 1, 4
   !          !DIR$ vector aligned
   !       do jj = 1, tile_size
   !         !DIR$ vector aligned
   !         do kk = 1, tile_size, 4
   !           !DIR$ vector aligned
   !           do ii = 1, tile_size
   !             dtmp_c_tiled(ii,jj,ll,m,j,a,k) =       dtmp_c_tiled(ii,jj,ll,m,j,a,k) +   &
   !                                   rescale_een_e_deriv_e_tiled(ii,kk+0,ll,j,i,k)   *   &
   !                                             rescale_een_n_tiled(kk+0,jj,m,i,a)    +   &
   !                                   rescale_een_e_deriv_e_tiled(ii,kk+1,ll,j,i,k)   *   &
   !                                             rescale_een_n_tiled(kk+1,jj,m,i,a)    +   &
   !                                   rescale_een_e_deriv_e_tiled(ii,kk+2,ll,j,i,k)   *   &
   !                                             rescale_een_n_tiled(kk+2,jj,m,i,a)    +   &
   !                                   rescale_een_e_deriv_e_tiled(ii,kk+3,ll,j,i,k)   *   &
   !                                             rescale_een_n_tiled(kk+3,jj,m,i,a)
   !         enddo
   !        enddo
   !       enddo
   !     enddo
   !    enddo
       enddo
     enddo
   enddo
 enddo
 !$OMP end do
 !$OMP end parallel 


END_PROVIDER

 BEGIN_PROVIDER [ double precision,  tmp_c_tiled_wj, (tile_size, tile_size,0:ncord, 0:ntiles_nelec, 0:ntiles_nnuc,0:ncord-1) ]
&BEGIN_PROVIDER [ double precision, dtmp_c_tiled_wj, (tile_size, tile_size,4,0:ncord, 0:ntiles_nelec, 0:ntiles_nnuc,0:ncord-1) ]
 !use tiling_interface
 implicit none
 BEGIN_DOC
 ! Calculate the intermediate buffers
 ! tmp_c:
 ! r_{ij}^k . R_{ja}^l -> tmp_c_{ia}^{kl}
 !
 ! dtmp_c:
 ! dr_{ij}^k . R_{ja}^l -> dtmp_c_{ia}^{kl}
 END_DOC
 integer :: k, i, j, a, l,m
 integer :: ii, jj, aa, kk, ll
 integer :: res

 ! r_{ij}^k . R_{ja}^l -> tmp_c_{ia}^{kl}
 !$OMP parallel                           &
 !$OMP private(m, jj, kk, ii, ll)         &
 !$OMP default (shared)                   
 !$OMP do collapse(3) schedule(dynamic)
 do k=0,ncord-1
   do j = 0, ntiles_nelec - 1
     do a = 0, ntiles_nnuc - 1
       do i = 0, ntiles_nelec - 1
   !call dgemm('N','N', tile_size, tile_size*(ncord+1), tile_size, 1.d0,           &
   !    rescale_een_e_tiled(1,1,j,i,k), size(rescale_een_e_tiled,1),                  &
   !    rescale_een_n_tiled(1,1,0,i,a), size(rescale_een_n_tiled,1), 1.d0,            &
   !    tmp_c_tiled(1,1,0,j,a,k), size(tmp_c_tiled,1))
   !call run_magma_dgemm_async_gpu_c(rescale_een_e_tiled(1,1,j,i,k),       &
   !                                rescale_een_n_tiled(1,1,0,i,a), &
   !                                tmp_c_tiled(1,1,0,j,a,k),       &
   !                                tile_size, tile_size*(ncord+1), &
   !                                tile_size,                      &
   !                                size(rescale_een_e_tiled,1),    &
   !                                size(rescale_een_n_tiled,1),    &
   !                                size(tmp_c_tiled,1))
        do m = 0, ncord
              !DIR$ vector aligned
          do jj = 1, tile_size, 4
            !DIR$ vector aligned
            do kk = 1, tile_size, 4
              !DIR$ UNROLL=24
              !DIR$ VECTOR
              !DIR$ vector aligned
              do ii = 1, tile_size
                 !tmp_c_tiled(ii,jj,m,j,a,k) = tmp_c_tiled(ii,jj,m,j,a,k) + &
                 !                         rescale_een_e_tiled(ii,kk+0,j,i,k)*&
                 !                         rescale_een_n_tiled(kk+0,jj,m,i,a)+&
                 !                         rescale_een_e_tiled(ii,kk+1,j,i,k)*&
                 !                         rescale_een_n_tiled(kk+1,jj,m,i,a)+&
                 !                         rescale_een_e_tiled(ii,kk+2,j,i,k)*&
                 !                         rescale_een_n_tiled(kk+2,jj,m,i,a)+&
                 !                         rescale_een_e_tiled(ii,kk+3,j,i,k)*&
                 !                         rescale_een_n_tiled(kk+3,jj,m,i,a)
                 tmp_c_tiled_wj(ii,jj+0,m,j,a,k) = tmp_c_tiled_wj(ii,jj+0,m,j,a,k) + &
                                          rescale_een_e_tiled(ii,kk+0,j,i,k)*&
                                          rescale_een_n_tiled(kk+0,jj+0,m,i,a)+&
                                          rescale_een_e_tiled(ii,kk+1,j,i,k)*&
                                          rescale_een_n_tiled(kk+1,jj+0,m,i,a)+&
                                          rescale_een_e_tiled(ii,kk+2,j,i,k)*&
                                          rescale_een_n_tiled(kk+2,jj+0,m,i,a)+&
                                          rescale_een_e_tiled(ii,kk+3,j,i,k)*&
                                          rescale_een_n_tiled(kk+3,jj+0,m,i,a)
                 tmp_c_tiled_wj(ii,jj+1,m,j,a,k) = tmp_c_tiled_wj(ii,jj+1,m,j,a,k) + &
                                          rescale_een_e_tiled(ii,kk+0,j,i,k)*&
                                          rescale_een_n_tiled(kk+0,jj+1,m,i,a)+&
                                          rescale_een_e_tiled(ii,kk+1,j,i,k)*&
                                          rescale_een_n_tiled(kk+1,jj+1,m,i,a)+&
                                          rescale_een_e_tiled(ii,kk+2,j,i,k)*&
                                          rescale_een_n_tiled(kk+2,jj+1,m,i,a)+&
                                          rescale_een_e_tiled(ii,kk+3,j,i,k)*&
                                          rescale_een_n_tiled(kk+3,jj+1,m,i,a)
                 tmp_c_tiled_wj(ii,jj+2,m,j,a,k) = tmp_c_tiled_wj(ii,jj+2,m,j,a,k) + &
                                          rescale_een_e_tiled(ii,kk+0,j,i,k)*&
                                          rescale_een_n_tiled(kk+0,jj+2,m,i,a)+&
                                          rescale_een_e_tiled(ii,kk+1,j,i,k)*&
                                          rescale_een_n_tiled(kk+1,jj+2,m,i,a)+&
                                          rescale_een_e_tiled(ii,kk+2,j,i,k)*&
                                          rescale_een_n_tiled(kk+2,jj+2,m,i,a)+&
                                          rescale_een_e_tiled(ii,kk+3,j,i,k)*&
                                          rescale_een_n_tiled(kk+3,jj+2,m,i,a)
                 tmp_c_tiled_wj(ii,jj+3,m,j,a,k) = tmp_c_tiled_wj(ii,jj+3,m,j,a,k) + &
                                          rescale_een_e_tiled(ii,kk+0,j,i,k)*&
                                          rescale_een_n_tiled(kk+0,jj+3,m,i,a)+&
                                          rescale_een_e_tiled(ii,kk+1,j,i,k)*&
                                          rescale_een_n_tiled(kk+1,jj+3,m,i,a)+&
                                          rescale_een_e_tiled(ii,kk+2,j,i,k)*&
                                          rescale_een_n_tiled(kk+2,jj+3,m,i,a)+&
                                          rescale_een_e_tiled(ii,kk+3,j,i,k)*&
                                          rescale_een_n_tiled(kk+3,jj+3,m,i,a)
            enddo
           enddo
         enddo
        enddo
       enddo
     enddo
   enddo
 enddo
 !$OMP end do

 ! dr_{ij}^k . R_{ja}^l -> dtmp_c_{ia}^{kl}
 !$OMP do collapse(3) schedule(dynamic)
 do k=0,ncord-1
   do a = 0, ntiles_nnuc - 1
     do j = 0, ntiles_nelec - 1
       do i = 0, ntiles_nelec - 1
   !call dgemm('N','N', 4*tile_size, tile_size*(ncord+1), tile_size, 1.d0,         &
   !    rescale_een_e_deriv_e_tiled(1,1,1,j,i,k), 4*size(rescale_een_e_deriv_e_tiled,1),&
   !    rescale_een_n_tiled(1,1,0,i,a), size(rescale_een_n_tiled,1), 1.d0,            &
   !    dtmp_c_tiled(1,1,1,0,j,a,k), 4*size(dtmp_c_tiled,1))

   !call run_magma_dgemm_async_gpu_c(rescale_een_e_deriv_e_tiled(1,1,1,k,j,i), &
   !                                rescale_een_n_tiled(1,1,0,i,a),            &
   !                                dtmp_c_tiled(1,1,1,0,j,a,k),               &
   !                                4*tile_size, tile_size*(ncord+1),          &
   !                                tile_size,                                 &
   !                                4*size(rescale_een_e_deriv_e_tiled,1),     &
   !                                size(rescale_een_n_tiled,1),               &
   !                                4*size(dtmp_c_tiled,1))
        do m = 0, ncord
              !DIR$ vector aligned
          do ll = 1, 4
              !DIR$ vector aligned
           do jj = 1, tile_size, 4
             !DIR$ vector aligned
             do kk = 1, tile_size, 4
               !DIR$ UNROLL=24
               !DIR$ VECTOR
               !DIR$ vector aligned
               do ii = 1, tile_size
                 !dtmp_c_tiled(ii,jj,ll,m,j,a,k) =       dtmp_c_tiled(ii,jj,ll,m,j,a,k) +   &
                 !                      rescale_een_e_deriv_e_tiled(ii,kk+0,ll,j,i,k)   *   &
                 !                                rescale_een_n_tiled(kk+0,jj,m,i,a)    +   &
                 !                      rescale_een_e_deriv_e_tiled(ii,kk+1,ll,j,i,k)   *   &
                 !                                rescale_een_n_tiled(kk+1,jj,m,i,a)    +   &
                 !                      rescale_een_e_deriv_e_tiled(ii,kk+2,ll,j,i,k)   *   &
                 !                                rescale_een_n_tiled(kk+2,jj,m,i,a)    +   &
                 !                      rescale_een_e_deriv_e_tiled(ii,kk+3,ll,j,i,k)   *   &
                 !                                rescale_een_n_tiled(kk+3,jj,m,i,a)
                 dtmp_c_tiled_wj(ii,jj,ll,m,j,a,k) =       dtmp_c_tiled_wj(ii,jj+0,ll,m,j,a,k) +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+0,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+0,jj+0,m,i,a)    +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+1,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+1,jj+0,m,i,a)    +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+2,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+2,jj+0,m,i,a)    +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+3,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+3,jj+0,m,i,a)
                 dtmp_c_tiled_wj(ii,jj+1,ll,m,j,a,k) =       dtmp_c_tiled_wj(ii,jj+1,ll,m,j,a,k) +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+0,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+0,jj+1,m,i,a)    +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+1,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+1,jj+1,m,i,a)    +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+2,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+2,jj+1,m,i,a)    +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+3,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+3,jj+1,m,i,a)
                 dtmp_c_tiled_wj(ii,jj+2,ll,m,j,a,k) =       dtmp_c_tiled_wj(ii,jj+2,ll,m,j,a,k) +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+0,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+0,jj+2,m,i,a)    +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+1,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+1,jj+2,m,i,a)    +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+2,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+2,jj+2,m,i,a)    +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+3,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+3,jj+2,m,i,a)
                 dtmp_c_tiled_wj(ii,jj+3,ll,m,j,a,k) =       dtmp_c_tiled_wj(ii,jj+3,ll,m,j,a,k) +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+0,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+0,jj+3,m,i,a)    +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+1,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+1,jj+3,m,i,a)    +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+2,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+2,jj+3,m,i,a)    +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+3,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+3,jj+3,m,i,a)
             enddo
            enddo
           enddo
         enddo
        enddo
       enddo
     enddo
   enddo
 enddo
 !$OMP end do
 !$OMP end parallel 


END_PROVIDER

subroutine gemm_simd(x1, x2, y, m0)
      use :: iso_c_binding
      use simd
      type(simd_real8), intent(in) :: x1, x2
      type(simd_real8), intent(out) :: y
      type(simd_mask8), intent(in) :: m0
      type(simd_real8) :: a
      type(simd_mask8) :: m1
      type(simd_int4) :: i, i_max
      real*8 :: temp
      integer(c_int) :: itemp
      integer :: ii, kk
      logical :: true_for_any
!$omp simd  aligned(x1,x2,y)
      do ii=0, 3
         temp = 0.0d0
         if( ii / 2 == 0) then
           do kk=0, 3
             temp = temp + x1%x(ii*4 + kk)
           end do
         else
           do kk=0, 3
             temp = temp + x2%x((ii-2)*4 + kk)
           end do
         endif
         y%x(ii) = y%x(ii+4) + temp
      end do
end subroutine foo_simd

 BEGIN_PROVIDER [ double precision,  tmp_c_tiled_simd, (tile_size, tile_size,0:ncord, 0:ntiles_nelec, 0:ntiles_nnuc,0:ncord-1) ]
&BEGIN_PROVIDER [ double precision, dtmp_c_tiled_simd, (tile_size, tile_size,4,0:ncord, 0:ntiles_nelec, 0:ntiles_nnuc,0:ncord-1) ]
 !use tiling_interface
 use simd
 implicit none
 BEGIN_DOC
 ! Calculate the intermediate buffers
 ! tmp_c:
 ! r_{ij}^k . R_{ja}^l -> tmp_c_{ia}^{kl}
 !
 ! dtmp_c:
 ! dr_{ij}^k . R_{ja}^l -> dtmp_c_{ia}^{kl}
 END_DOC
 integer :: k, i, j, a, l,m
 integer :: ii, iii, jj, aa, kk, ll
 integer :: res
 type(simd_real8) :: buffer_x1, buffer_x2, buffer_y
 type(simd_mask8) :: msk


 ! r_{ij}^k . R_{ja}^l -> tmp_c_{ia}^{kl}
 !$OMP parallel                               &
 !$OMP private(m, msk, buffer_x1, buffer_x2, buffer_y, jj,iii,  kk, ii, ll) &
 !$OMP default (shared)
 !$OMP do collapse(3) schedule(dynamic)
 do k=0,ncord-1
   do j = 0, ntiles_nelec - 1
     do a = 0, ntiles_nnuc - 1
       do i = 0, ntiles_nelec - 1
        do m = 0, ncord
 !DIR$ vector aligned
         do jj = 1, tile_size, 4
 !DIR$ vector aligned
           do kk = 1, tile_size, 4
 !DIR$ UNROLL=24
 !DIR$ VECTOR
 !DIR$ vector aligned
             do ii = 1, tile_size
 !$OMP SIMD
                do iii=0, 7
                   msk%x(iii) = .true.
                   if ( iii / 4 .EQ. 1 ) then
                     buffer_x1%x(iii) = rescale_een_n_tiled(kk+iii,jj+0,m,i,a)
                     buffer_x2%x(iii) = rescale_een_n_tiled(kk+iii,jj+2,m,i,a)
                     buffer_y%x(iii)  = rescale_een_e_tiled(ii,kk+iii,j,i,k)
                   else
                     buffer_x1%x(iii) = rescale_een_n_tiled(kk+iii,jj+1,m,i,a)
                     buffer_x2%x(iii) = rescale_een_n_tiled(kk+iii,jj+3,m,i,a)
                     buffer_y%x(iii)  = tmp_c_tiled_simd(ii,jj+iii,m,j,a,k)
                   endif
                enddo
 !DIR$ NOINLINE
                call gemm_simd(buffer_x1, buffer_x2, buffer_y, msk)
 !$OMP SIMD
                 do iii=0, 3
                    tmp_c_tiled_simd(ii,jj+iii,m,j,a,k) = buffer_y%x(iii)
                 enddo
            enddo
           enddo
         enddo
        enddo
       enddo
     enddo
   enddo
 enddo
 !$OMP end do

 ! dr_{ij}^k . R_{ja}^l -> dtmp_c_{ia}^{kl}
 !$OMP do collapse(3) schedule(dynamic)
 do k=0,ncord-1
   do a = 0, ntiles_nnuc - 1
     do j = 0, ntiles_nelec - 1
       do i = 0, ntiles_nelec - 1
        do m = 0, ncord
 !DIR$ vector aligned
          do ll = 1, 4
 !DIR$ vector aligned
           do jj = 1, tile_size, 4
 !DIR$ vector aligned
             do kk = 1, tile_size, 4
 !DIR$ UNROLL=24
 !DIR$ VECTOR
 !DIR$ vector aligned
               do ii = 1, tile_size
                 dtmp_c_tiled_wj(ii,jj,ll,m,j,a,k) =       dtmp_c_tiled_wj(ii,jj+0,ll,m,j,a,k) +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+0,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+0,jj+0,m,i,a)    +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+1,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+1,jj+0,m,i,a)    +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+2,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+2,jj+0,m,i,a)    +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+3,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+3,jj+0,m,i,a)
                 dtmp_c_tiled_wj(ii,jj+1,ll,m,j,a,k) =       dtmp_c_tiled_wj(ii,jj+1,ll,m,j,a,k) +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+0,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+0,jj+1,m,i,a)    +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+1,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+1,jj+1,m,i,a)    +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+2,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+2,jj+1,m,i,a)    +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+3,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+3,jj+1,m,i,a)
                 dtmp_c_tiled_wj(ii,jj+2,ll,m,j,a,k) =       dtmp_c_tiled_wj(ii,jj+2,ll,m,j,a,k) +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+0,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+0,jj+2,m,i,a)    +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+1,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+1,jj+2,m,i,a)    +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+2,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+2,jj+2,m,i,a)    +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+3,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+3,jj+2,m,i,a)
                 dtmp_c_tiled_wj(ii,jj+3,ll,m,j,a,k) =       dtmp_c_tiled_wj(ii,jj+3,ll,m,j,a,k) +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+0,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+0,jj+3,m,i,a)    +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+1,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+1,jj+3,m,i,a)    +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+2,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+2,jj+3,m,i,a)    +   &
                                       rescale_een_e_deriv_e_tiled(ii,kk+3,ll,j,i,k)   *   &
                                                 rescale_een_n_tiled(kk+3,jj+3,m,i,a)
             enddo
            enddo
           enddo
         enddo
        enddo
       enddo
     enddo
   enddo
 enddo
 !$OMP end do
 !$OMP end parallel 


END_PROVIDER

