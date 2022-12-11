# mark_description "Intel(R) C++ Intel(R) 64 Compiler for applications running on Intel(R) 64, Version 19.1.1.217 Build 20200306";
# mark_description "";
# mark_description "-std=c++11 -lpapi -ansi-alias -O2 -Wall -DN=1000 -DDT=0.001f -DSTEPS=1000 -S -fsource-asm -c";
	.file "nbody.cpp"
	.text
..TXTST0:
.L_2__routine_start__Z18particles_simulateRA1000_10t_particle_0:
# -- Begin  _Z18particles_simulateRA1000_10t_particle
	.text
# mark_begin;
       .align    16,0x90
	.globl _Z18particles_simulateRA1000_10t_particle
# --- particles_simulate(t_particles &)
_Z18particles_simulateRA1000_10t_particle:
# parameter 1: %rdi
..B1.1:                         # Preds ..B1.0
                                # Execution count [1.00e+00]

### {

	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
..___tag_value__Z18particles_simulateRA1000_10t_particle.1:
..L2:
                                                          #10.1
        pushq     %rbp                                          #10.1
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
        subq      $12080, %rsp                                  #10.1
	.cfi_def_cfa_offset 12096
        movq      %rdi, %rbp                                    #10.1

###     int i, j, k;
### 
###     t_velocities velocities = {};

        xorl      %esi, %esi                                    #13.29
        lea       48(%rsp), %rdi                                #13.29
        movl      $12000, %edx                                  #13.29
        call      _intel_fast_memset                            #13.29
                                # LOE rbx rbp r12 r13 r14 r15
..B1.2:                         # Preds ..B1.1
                                # Execution count [0.00e+00]

### 
###     for (k = 0; k < STEPS; k++)

        xorl      %ecx, %ecx                                    #15.10

###     {
###             //vynulovani mezisouctu
###         for (i = 0; i < N; i++)
###         {
###             velocities[i].x = 0.0f;
###             velocities[i].y = 0.0f;
###             velocities[i].z = 0.0f;
###         }
###             //vypocet nove rychlosti
###         for (i = 0; i < N; i++)
###         {
###             for (j = 0; j < N; j++)
###             {
###                 calculate_gravitation_velocity(p[j], p[i], velocities[j]);
###                 calculate_collision_velocity(p[j], p[i], velocities[j]);
###             }
###         }
###             //ulozeni rychlosti a posun castic
###         for (i = 0; i < N; i++)
###         {
###             p[i].vel_x += velocities[i].x;
###             p[i].vel_y += velocities[i].y;
###             p[i].vel_z += velocities[i].z;
### 
###             p[i].pos_x += p[i].vel_x * DT;
###             p[i].pos_y += p[i].vel_y * DT;
###             p[i].pos_z += p[i].vel_z * DT;

        xorl      %eax, %eax                                    #42.40
        movss     .L_2il0floatpacket.12(%rip), %xmm0            #42.40
        movq      %r12, (%rsp)                                  #42.40[spill]
	.cfi_offset 12, -12096
        movl      %ecx, %r12d                                   #42.40
        movq      %r13, 8(%rsp)                                 #42.40[spill]
	.cfi_offset 13, -12088
        movl      %ecx, %r13d                                   #42.40
        movq      %r14, 16(%rsp)                                #42.40[spill]
        movq      %r15, 24(%rsp)                                #42.40[spill]
        movq      %rbx, 32(%rsp)                                #42.40[spill]
	.cfi_offset 3, -12064
	.cfi_offset 14, -12080
	.cfi_offset 15, -12072
        movq      %rbp, %rbx                                    #42.40
        movq      %rax, %rbp                                    #42.40
                                # LOE rbx rbp r12d r13d
..B1.3:                         # Preds ..B1.12 ..B1.2
                                # Execution count [5.00e+00]
        xorl      %esi, %esi                                    #18.9
        lea       48(%rsp), %rdi                                #18.9
        movl      $12000, %edx                                  #18.9
        call      _intel_fast_memset                            #18.9
                                # LOE rbx rbp r12d r13d
..B1.4:                         # Preds ..B1.3
                                # Execution count [1.00e+03]
        movl      %r13d, 40(%rsp)                               #25.14[spill]
        movl      %r12d, %edi                                   #25.14
        movq      %rbp, %rsi                                    #25.14
                                # LOE rbx rbp rsi edi r12d
..B1.5:                         # Preds ..B1.9 ..B1.4
                                # Execution count [1.00e+06]
        movq      %rbp, %r13                                    #27.18
        lea       (%rbx,%rsi), %rcx                             #29.54
        movq      %rcx, 12064(%rsp)                             #27.18[spill]
        movl      %r12d, %r15d                                  #27.18
        movq      %rsi, 12056(%rsp)                             #27.18[spill]
        movq      %r13, %r14                                    #27.18
        movl      %edi, 12048(%rsp)                             #27.18[spill]
                                # LOE rbx r13 r14 r15d
..B1.6:                         # Preds ..B1.8 ..B1.5
                                # Execution count [1.00e+09]
        movq      12064(%rsp), %rsi                             #29.17[spill]
        lea       (%rbx,%r13), %r12                             #29.48
        movq      %r12, %rdi                                    #29.17
        lea       48(%rsp,%r14), %rbp                           #29.60
        movq      %rbp, %rdx                                    #29.17
..___tag_value__Z18particles_simulateRA1000_10t_particle.21:
#       calculate_gravitation_velocity(const t_particle &, const t_particle &, t_velocity &)
        call      _Z30calculate_gravitation_velocityRK10t_particleS1_R10t_velocity #29.17
..___tag_value__Z18particles_simulateRA1000_10t_particle.22:
                                # LOE rbx rbp r12 r13 r14 r15d
..B1.7:                         # Preds ..B1.6
                                # Execution count [1.00e+09]
        movq      %r12, %rdi                                    #30.17
        movq      %rbp, %rdx                                    #30.17
        movq      12064(%rsp), %rsi                             #30.17[spill]
..___tag_value__Z18particles_simulateRA1000_10t_particle.23:
#       calculate_collision_velocity(const t_particle &, const t_particle &, t_velocity &)
        call      _Z28calculate_collision_velocityRK10t_particleS1_R10t_velocity #30.17
..___tag_value__Z18particles_simulateRA1000_10t_particle.24:
                                # LOE rbx r13 r14 r15d
..B1.8:                         # Preds ..B1.7
                                # Execution count [1.00e+09]
        incl      %r15d                                         #27.32
        addq      $28, %r13                                     #27.32
        addq      $12, %r14                                     #27.32
        cmpl      $1000, %r15d                                  #27.29
        jl        ..B1.6        # Prob 99%                      #27.29
                                # LOE rbx r13 r14 r15d
..B1.9:                         # Preds ..B1.8
                                # Execution count [1.00e+06]
        movl      12048(%rsp), %edi                             #[spill]
        xorl      %ebp, %ebp                                    #
        incl      %edi                                          #25.28
        xorl      %r12d, %r12d                                  #
        movq      12056(%rsp), %rsi                             #[spill]
        addq      $28, %rsi                                     #25.28
        cmpl      $1000, %edi                                   #25.25
        jl        ..B1.5        # Prob 99%                      #25.25
                                # LOE rbx rbp rsi edi r12d
..B1.10:                        # Preds ..B1.9
                                # Execution count [1.00e+03]
        movq      %rbp, %rdi                                    #34.9
        movl      %r12d, %r8d                                   #34.9
        movl      40(%rsp), %r13d                               #[spill]
        movq      %rdi, %rsi                                    #34.9
        movss     .L_2il0floatpacket.12(%rip), %xmm3            #34.9
        .align    16,0x90
                                # LOE rbx rbp rsi rdi r8d r12d r13d xmm3
..B1.11:                        # Preds ..B1.11 ..B1.10
                                # Execution count [1.00e+06]
        movss     20(%rdi,%rbx), %xmm0                          #38.13
        incl      %r8d                                          #34.9
        movss     16(%rdi,%rbx), %xmm1                          #37.13
        movss     12(%rdi,%rbx), %xmm2                          #36.13
        addss     56(%rsp,%rsi), %xmm0                          #38.13
        addss     52(%rsp,%rsi), %xmm1                          #37.13
        addss     48(%rsp,%rsi), %xmm2                          #36.13
        movss     %xmm0, 20(%rdi,%rbx)                          #38.13
        addq      $12, %rsi                                     #34.9
        movss     %xmm1, 16(%rdi,%rbx)                          #37.13
        movss     %xmm2, 12(%rdi,%rbx)                          #36.13
        mulss     %xmm3, %xmm0                                  #42.40
        mulss     %xmm3, %xmm1                                  #41.40
        mulss     %xmm3, %xmm2                                  #40.40
        addss     8(%rdi,%rbx), %xmm0                           #42.13
        addss     4(%rdi,%rbx), %xmm1                           #41.13
        addss     (%rdi,%rbx), %xmm2                            #40.13
        movss     %xmm0, 8(%rdi,%rbx)                           #42.13
        movss     %xmm1, 4(%rdi,%rbx)                           #41.13
        movss     %xmm2, (%rdi,%rbx)                            #40.13
        addq      $28, %rdi                                     #34.9
        cmpl      $1000, %r8d                                   #34.9
        jb        ..B1.11       # Prob 99%                      #34.9
                                # LOE rbx rbp rsi rdi r8d r12d r13d xmm3
..B1.12:                        # Preds ..B1.11
                                # Execution count [1.00e+03]
        incl      %r13d                                         #15.28
        cmpl      $1000, %r13d                                  #15.21
        jl        ..B1.3        # Prob 99%                      #15.21
                                # LOE rbx rbp r12d r13d
..B1.13:                        # Preds ..B1.12
                                # Execution count [1.00e+00]
        movq      (%rsp), %r12                                  #[spill]
	.cfi_restore 12
        movq      8(%rsp), %r13                                 #[spill]
	.cfi_restore 13
        movq      16(%rsp), %r14                                #[spill]
	.cfi_restore 14
        movq      24(%rsp), %r15                                #[spill]
	.cfi_restore 15
        movq      32(%rsp), %rbx                                #[spill]
	.cfi_restore 3

###         }
###     }
### }

        addq      $12080, %rsp                                  #45.1
	.cfi_def_cfa_offset 16
	.cfi_restore 6
        popq      %rbp                                          #45.1
	.cfi_def_cfa_offset 8
        ret                                                     #45.1
        .align    16,0x90
                                # LOE
	.cfi_endproc
# mark_end;
	.type	_Z18particles_simulateRA1000_10t_particle,@function
	.size	_Z18particles_simulateRA1000_10t_particle,.-_Z18particles_simulateRA1000_10t_particle
..LN_Z18particles_simulateRA1000_10t_particle.0:
	.data
# -- End  _Z18particles_simulateRA1000_10t_particle
	.text
.L_2__routine_start__Z14particles_readP8_IO_FILERA1000_10t_particle_1:
# -- Begin  _Z14particles_readP8_IO_FILERA1000_10t_particle
	.text
# mark_begin;
       .align    16,0x90
	.globl _Z14particles_readP8_IO_FILERA1000_10t_particle
# --- particles_read(FILE *, t_particles &)
_Z14particles_readP8_IO_FILERA1000_10t_particle:
# parameter 1: %rdi
# parameter 2: %rsi
..B2.1:                         # Preds ..B2.0
                                # Execution count [1.00e+00]

### {

	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
..___tag_value__Z14particles_readP8_IO_FILERA1000_10t_particle.42:
..L43:
                                                         #49.1
        pushq     %r12                                          #49.1
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
        pushq     %r13                                          #49.1
	.cfi_def_cfa_offset 24
	.cfi_offset 13, -24
        pushq     %r14                                          #49.1
	.cfi_def_cfa_offset 32
	.cfi_offset 14, -32

###     for (int i = 0; i < N; i++)

        xorl      %eax, %eax                                    #50.16
        movl      %eax, %r12d                                   #50.16
        movq      %rsi, %r14                                    #50.16
        movq      %rdi, %r13                                    #50.16
                                # LOE rbx rbp r13 r14 r15 r12d
..B2.2:                         # Preds ..B2.3 ..B2.1
                                # Execution count [1.00e+03]

###     {
###         fscanf(fp, "%f %f %f %f %f %f %f \n",

        addq      $-32, %rsp                                    #52.9
	.cfi_def_cfa_offset 64
        lea       4(%r14), %rcx                                 #52.9
        movq      %r13, %rdi                                    #52.9
        lea       8(%r14), %r8                                  #52.9
        movl      $.L_2__STRING.0, %esi                         #52.9
        lea       12(%r14), %r9                                 #52.9
        movq      %r14, %rdx                                    #52.9
        xorl      %eax, %eax                                    #52.9
        lea       16(%r14), %r10                                #52.9
        movq      %r10, (%rsp)                                  #52.9
        lea       20(%r14), %r11                                #52.9
        movq      %r11, 8(%rsp)                                 #52.9
        lea       24(%r14), %r10                                #52.9
        movq      %r10, 16(%rsp)                                #52.9
#       fscanf(FILE *, const char *, ...)
        call      __isoc99_fscanf                               #52.9
                                # LOE rbx rbp r13 r14 r15 r12d
..B2.7:                         # Preds ..B2.2
                                # Execution count [1.00e+03]
        addq      $32, %rsp                                     #52.9
	.cfi_def_cfa_offset 32
                                # LOE rbx rbp r13 r14 r15 r12d
..B2.3:                         # Preds ..B2.7
                                # Execution count [1.00e+03]
        incl      %r12d                                         #50.28
        addq      $28, %r14                                     #50.28
        cmpl      $1000, %r12d                                  #50.25
        jl        ..B2.2        # Prob 99%                      #50.25
                                # LOE rbx rbp r13 r14 r15 r12d
..B2.4:                         # Preds ..B2.3
                                # Execution count [1.00e+00]

###             &p[i].pos_x, &p[i].pos_y, &p[i].pos_z,
###             &p[i].vel_x, &p[i].vel_y, &p[i].vel_z,
###             &p[i].weight);
###     }
### }

	.cfi_restore 14
        popq      %r14                                          #57.1
	.cfi_def_cfa_offset 24
	.cfi_restore 13
        popq      %r13                                          #57.1
	.cfi_def_cfa_offset 16
	.cfi_restore 12
        popq      %r12                                          #57.1
	.cfi_def_cfa_offset 8
        ret                                                     #57.1
        .align    16,0x90
                                # LOE
	.cfi_endproc
# mark_end;
	.type	_Z14particles_readP8_IO_FILERA1000_10t_particle,@function
	.size	_Z14particles_readP8_IO_FILERA1000_10t_particle,.-_Z14particles_readP8_IO_FILERA1000_10t_particle
..LN_Z14particles_readP8_IO_FILERA1000_10t_particle.1:
	.data
# -- End  _Z14particles_readP8_IO_FILERA1000_10t_particle
	.text
.L_2__routine_start__Z15particles_writeP8_IO_FILERA1000_10t_particle_2:
# -- Begin  _Z15particles_writeP8_IO_FILERA1000_10t_particle
	.text
# mark_begin;
       .align    16,0x90
	.globl _Z15particles_writeP8_IO_FILERA1000_10t_particle
# --- particles_write(FILE *, t_particles &)
_Z15particles_writeP8_IO_FILERA1000_10t_particle:
# parameter 1: %rdi
# parameter 2: %rsi
..B3.1:                         # Preds ..B3.0
                                # Execution count [1.00e+00]

### {

	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
..___tag_value__Z15particles_writeP8_IO_FILERA1000_10t_particle.59:
..L60:
                                                         #60.1
        pushq     %r12                                          #60.1
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
        pushq     %r13                                          #60.1
	.cfi_def_cfa_offset 24
	.cfi_offset 13, -24
        pushq     %r14                                          #60.1
	.cfi_def_cfa_offset 32
	.cfi_offset 14, -32
        pushq     %r15                                          #60.1
	.cfi_def_cfa_offset 40
	.cfi_offset 15, -40
        pushq     %rsi                                          #60.1
	.cfi_def_cfa_offset 48

###     for (int i = 0; i < N; i++)

        xorl      %edx, %edx                                    #61.16
        xorl      %eax, %eax                                    #61.16
        movl      %edx, %r13d                                   #61.16
        movq      %rax, %r12                                    #61.16
        movq      %rsi, %r15                                    #61.16
        movq      %rdi, %r14                                    #61.16
                                # LOE rbx rbp r12 r14 r15 r13d
..B3.2:                         # Preds ..B3.3 ..B3.1
                                # Execution count [1.00e+03]

###     {
###         fprintf(fp, "%10.10f %10.10f %10.10f %10.10f %10.10f %10.10f %10.10f \n",

        pxor      %xmm0, %xmm0                                  #63.9
        pxor      %xmm1, %xmm1                                  #63.9
        pxor      %xmm2, %xmm2                                  #63.9
        pxor      %xmm3, %xmm3                                  #63.9
        pxor      %xmm4, %xmm4                                  #63.9
        pxor      %xmm5, %xmm5                                  #63.9
        pxor      %xmm6, %xmm6                                  #63.9
        movq      %r14, %rdi                                    #63.9
        cvtss2sd  (%r12,%r15), %xmm0                            #63.9
        cvtss2sd  4(%r12,%r15), %xmm1                           #63.9
        cvtss2sd  8(%r12,%r15), %xmm2                           #63.9
        cvtss2sd  12(%r12,%r15), %xmm3                          #63.9
        cvtss2sd  16(%r12,%r15), %xmm4                          #63.9
        cvtss2sd  20(%r12,%r15), %xmm5                          #63.9
        cvtss2sd  24(%r12,%r15), %xmm6                          #63.9
        movl      $.L_2__STRING.1, %esi                         #63.9
        movl      $7, %eax                                      #63.9
#       fprintf(FILE *, const char *, ...)
        call      fprintf                                       #63.9
                                # LOE rbx rbp r12 r14 r15 r13d
..B3.3:                         # Preds ..B3.2
                                # Execution count [1.00e+03]
        incl      %r13d                                         #61.28
        addq      $28, %r12                                     #61.28
        cmpl      $1000, %r13d                                  #61.25
        jl        ..B3.2        # Prob 99%                      #61.25
                                # LOE rbx rbp r12 r14 r15 r13d
..B3.4:                         # Preds ..B3.3
                                # Execution count [1.00e+00]

###             p[i].pos_x, p[i].pos_y, p[i].pos_z,
###             p[i].vel_x, p[i].vel_y, p[i].vel_z,
###             p[i].weight);
###     }
### }

        popq      %rcx                                          #68.1
	.cfi_def_cfa_offset 40
	.cfi_restore 15
        popq      %r15                                          #68.1
	.cfi_def_cfa_offset 32
	.cfi_restore 14
        popq      %r14                                          #68.1
	.cfi_def_cfa_offset 24
	.cfi_restore 13
        popq      %r13                                          #68.1
	.cfi_def_cfa_offset 16
	.cfi_restore 12
        popq      %r12                                          #68.1
	.cfi_def_cfa_offset 8
        ret                                                     #68.1
        .align    16,0x90
                                # LOE
	.cfi_endproc
# mark_end;
	.type	_Z15particles_writeP8_IO_FILERA1000_10t_particle,@function
	.size	_Z15particles_writeP8_IO_FILERA1000_10t_particle,.-_Z15particles_writeP8_IO_FILERA1000_10t_particle
..LN_Z15particles_writeP8_IO_FILERA1000_10t_particle.2:
	.data
# -- End  _Z15particles_writeP8_IO_FILERA1000_10t_particle
	.section .rodata, "a"
	.align 4
	.align 4
.L_2il0floatpacket.12:
	.long	0x3a83126f
	.type	.L_2il0floatpacket.12,@object
	.size	.L_2il0floatpacket.12,4
	.section .rodata.str1.4, "aMS",@progbits,1
	.align 4
	.align 4
.L_2__STRING.0:
	.long	622880293
	.long	1713709158
	.long	543565088
	.long	622880293
	.long	1713709158
	.word	2592
	.byte	0
	.type	.L_2__STRING.0,@object
	.size	.L_2__STRING.0,23
	.space 1, 0x00 	# pad
	.align 4
.L_2__STRING.1:
	.long	774910245
	.long	543567921
	.long	774910245
	.long	543567921
	.long	774910245
	.long	543567921
	.long	774910245
	.long	543567921
	.long	774910245
	.long	543567921
	.long	774910245
	.long	543567921
	.long	774910245
	.long	543567921
	.word	10
	.type	.L_2__STRING.1,@object
	.size	.L_2__STRING.1,58
	.data
	.section .note.GNU-stack, ""
# End
