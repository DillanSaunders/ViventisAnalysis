using batch_quickpiv

#this julia file is the one run by python



batch_piv(ARGS[1],ARGS[2],parse(Bool,ARGS[3]),parse(Int,ARGS[4]),parse(Int,ARGS[5]),parse(Int,ARGS[6]),Tuple(Meta.parse(ARGS[7]).args))

#batch_quickpiv.batch_piv("Z:\\Room224_SharedFolder\\Alice\\pescoid\\20250316_153558_expzacy0047_fused\\","6_Settings1.h5",true,0,40,20,(0.4,0.4,2))