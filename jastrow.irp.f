program jastrow
  implicit none
  print *, 'The total Jastrow factor'
  print *, jastrow_full
  print *, 'REF'
  print *, factor_een_deriv_e_ref
  print *, 'X'
  print *, factor_ee_deriv_e
  print *, factor_een_deriv_e
  print *, 'BLAS'
  print *, factor_een_deriv_e_blas_tiled
  !PROVIDE jastrow_full

end program
