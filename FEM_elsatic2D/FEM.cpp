/*
    Concepts: KSP^solving a system of linear equations
    Concepts: KSP^Laplacian, 3d
    Processors: n
 */


static char help[] = "Solves the compressible plane strain elasticity equations in 2d on the unit domain using Q1 finite elements\n\n";

#include <iostream>
#include <petscksp.h>
#include <petscdmda.h>
#include <petscdm.h>

static PetscErrorCode DMDABCApplyCompression(DM,Mat,Vec);


#define NSD            2 /* number of spatial dimensions */
#define NODES_PER_EL   4 /* nodes per element */
#define U_DOFS         2
#define GAUSS_POINTS   4

/* cell based evaluation */
typedef struct {
    PetscScalar E,nu,fx,fy;
} Coefficients;

/* Gauss point based evaluation 8+4+4+4 = 20 */
typedef struct {
    PetscScalar gp_coords[2*GAUSS_POINTS];
    PetscScalar E[GAUSS_POINTS];
    PetscScalar nu[GAUSS_POINTS];
    PetscScalar fx[GAUSS_POINTS];
    PetscScalar fy[GAUSS_POINTS];
} GaussPointCoefficients;

typedef struct {
    PetscScalar ux_dof;
    PetscScalar uy_dof;
} ElasticityDOF;

/*

 D = E/((1+nu)(1-2nu)) * [ 1-nu   nu        0     ]
                         [  nu   1-nu       0     ]
                         [  0     0   0.5*(1-2nu) ]

 B = [ d_dx   0   ]
     [  0    d_dy ]
     [ d_dy  d_dx ]

 */


/* FEM routines */
/*
 Element: Local basis function ordering
 1-----2
 |     |
 |     |
 0-----3
 */

static void ConstructQ12D_Ni(PetscScalar _xi[],PetscScalar Ni[])
{
    PetscScalar xi  = _xi[0];
    PetscScalar eta = _xi[1];

    Ni[0] = 0.25*(1.0-xi)*(1.0-eta);
    Ni[1] = 0.25*(1.0-xi)*(1.0+eta);
    Ni[2] = 0.25*(1.0+xi)*(1.0+eta);
    Ni[3] = 0.25*(1.0+xi)*(1.0-eta);
}

static void ConstructQ12D_GNi(PetscScalar _xi[],PetscScalar GNi[][NODES_PER_EL])
{
    PetscScalar xi  = _xi[0];
    PetscScalar eta = _xi[1];

    GNi[0][0] = -0.25*(1.0-eta);
    GNi[0][1] = -0.25*(1.0+eta);
    GNi[0][2] =  0.25*(1.0+eta);
    GNi[0][3] =  0.25*(1.0-eta);

    GNi[1][0] = -0.25*(1.0-xi);
    GNi[1][1] =  0.25*(1.0-xi);
    GNi[1][2] =  0.25*(1.0+xi);
    GNi[1][3] = -0.25*(1.0+xi);
}

static void ConstructQ12D_GNx(PetscScalar GNi[][NODES_PER_EL],PetscScalar GNx[][NODES_PER_EL],PetscScalar coords[],PetscScalar *det_J)
{
    PetscScalar J00,J01,J10,J11,J;
    PetscScalar iJ00,iJ01,iJ10,iJ11;
    PetscInt    i;

    J00 = J01 = J10 = J11 = 0.0;
    for (i = 0; i < NODES_PER_EL; i++) {
        PetscScalar cx = coords[2*i+0];
        PetscScalar cy = coords[2*i+1];

        J00 = J00+GNi[0][i]*cx;      /* J_xx = dx/dxi */
        J01 = J01+GNi[0][i]*cy;      /* J_xy = dy/dxi */
        J10 = J10+GNi[1][i]*cx;      /* J_yx = dx/deta */
        J11 = J11+GNi[1][i]*cy;      /* J_yy = dy/deta */
    }
    J = (J00*J11)-(J01*J10);

    iJ00 =  J11/J;
    iJ01 = -J01/J;
    iJ10 = -J10/J;
    iJ11 =  J00/J;


    for (i = 0; i < NODES_PER_EL; i++) {
        GNx[0][i] = GNi[0][i]*iJ00+GNi[1][i]*iJ01;
        GNx[1][i] = GNi[0][i]*iJ10+GNi[1][i]*iJ11;
    }

    if (det_J) *det_J = J;
}


static void ConstructGaussQuadrature(PetscInt *ngp,PetscScalar gp_xi[][2],PetscScalar gp_weight[])
{
    *ngp         = 4;
    gp_xi[0][0]  = -0.57735026919;gp_xi[0][1] = -0.57735026919;
    gp_xi[1][0]  = -0.57735026919;gp_xi[1][1] =  0.57735026919;
    gp_xi[2][0]  =  0.57735026919;gp_xi[2][1] =  0.57735026919;
    gp_xi[3][0]  =  0.57735026919;gp_xi[3][1] = -0.57735026919;
    gp_weight[0] = 1.0;
    gp_weight[1] = 1.0;
    gp_weight[2] = 1.0;
    gp_weight[3] = 1.0;
}

static PetscErrorCode DMDAGetElementOwnershipRanges2d(DM da,PetscInt **_lx,PetscInt **_ly)
{
    PetscErrorCode ierr;
    PetscMPIInt    rank;
    PetscInt       proc_I,proc_J;
    PetscInt       cpu_x,cpu_y;
    PetscInt       local_mx,local_my;
    Vec            vlx,vly;
    PetscInt       *LX,*LY,i;
    PetscScalar    *_a;
    Vec            V_SEQ;
    VecScatter     ctx;

    PetscFunctionBeginUser;
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

    DMDAGetInfo(da,0,0,0,0,&cpu_x,&cpu_y,0,0,0,0,0,0,0);

    proc_J = rank/cpu_x;
    proc_I = rank-cpu_x*proc_J;

    ierr = PetscMalloc1(cpu_x,&LX);CHKERRQ(ierr);
    ierr = PetscMalloc1(cpu_y,&LY);CHKERRQ(ierr);

    ierr = DMDAGetElementsSizes(da,&local_mx,&local_my,NULL);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_WORLD,&vlx);CHKERRQ(ierr);
    ierr = VecSetSizes(vlx,PETSC_DECIDE,cpu_x);CHKERRQ(ierr);
    ierr = VecSetFromOptions(vlx);CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD,&vly);CHKERRQ(ierr);
    ierr = VecSetSizes(vly,PETSC_DECIDE,cpu_y);CHKERRQ(ierr);
    ierr = VecSetFromOptions(vly);CHKERRQ(ierr);

    ierr = VecSetValue(vlx,proc_I,(PetscScalar)(local_mx+1.0e-9),INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(vly,proc_J,(PetscScalar)(local_my+1.0e-9),INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(vlx);VecAssemblyEnd(vlx);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(vly);VecAssemblyEnd(vly);CHKERRQ(ierr);

    ierr = VecScatterCreateToAll(vlx,&ctx,&V_SEQ);CHKERRQ(ierr);
    ierr = VecScatterBegin(ctx,vlx,V_SEQ,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ctx,vlx,V_SEQ,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray(V_SEQ,&_a);CHKERRQ(ierr);
    for (i = 0; i < cpu_x; i++) LX[i] = (PetscInt)PetscRealPart(_a[i]);
    ierr = VecRestoreArray(V_SEQ,&_a);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);
    ierr = VecDestroy(&V_SEQ);CHKERRQ(ierr);

    ierr = VecScatterCreateToAll(vly,&ctx,&V_SEQ);CHKERRQ(ierr);
    ierr = VecScatterBegin(ctx,vly,V_SEQ,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ctx,vly,V_SEQ,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray(V_SEQ,&_a);CHKERRQ(ierr);
    for (i = 0; i < cpu_y; i++) LY[i] = (PetscInt)PetscRealPart(_a[i]);
    ierr = VecRestoreArray(V_SEQ,&_a);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);
    ierr = VecDestroy(&V_SEQ);CHKERRQ(ierr);

    *_lx = LX;
    *_ly = LY;

    ierr = VecDestroy(&vlx);CHKERRQ(ierr);
    ierr = VecDestroy(&vly);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

static PetscErrorCode DMDAViewGnuplot2d(DM da,Vec fields,const char comment[],const char prefix[])
{
    DM             cda;
    Vec            coords,local_fields;
    DMDACoor2d     **_coords;
    FILE           *fp;
    char           fname[PETSC_MAX_PATH_LEN];
    const char     *field_name;
    PetscMPIInt    rank;
    PetscInt       si,sj,nx,ny,i,j;
    PetscInt       n_dofs,d;
    PetscScalar    *_fields;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    ierr = PetscSNPrintf(fname,sizeof(fname),"%s-p%1.4d.dat",prefix,rank);CHKERRQ(ierr);
    ierr = PetscFOpen(PETSC_COMM_SELF,fname,"w",&fp);CHKERRQ(ierr);
    if (!fp) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot open file");

    ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"### %s (processor %1.4d) ### \n",comment,rank);CHKERRQ(ierr);
    ierr = DMDAGetInfo(da,0,0,0,0,0,0,0,&n_dofs,0,0,0,0,0);CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"### x y ");CHKERRQ(ierr);
    for (d = 0; d < n_dofs; d++) {
        ierr = DMDAGetFieldName(da,d,&field_name);CHKERRQ(ierr);
        ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%s ",field_name);CHKERRQ(ierr);
    }
    ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"###\n");CHKERRQ(ierr);


    ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(da,&coords);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(cda,coords,&_coords);CHKERRQ(ierr);
    ierr = DMDAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0);CHKERRQ(ierr);

    ierr = DMCreateLocalVector(da,&local_fields);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da,fields,INSERT_VALUES,local_fields);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da,fields,INSERT_VALUES,local_fields);CHKERRQ(ierr);
    ierr = VecGetArray(local_fields,&_fields);CHKERRQ(ierr);

    for (j = sj; j < sj+ny; j++) {
        for (i = si; i < si+nx; i++) {
            PetscScalar coord_x,coord_y;
            PetscScalar field_d;

            coord_x = _coords[j][i].x;
            coord_y = _coords[j][i].y;

            ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e %1.6e ",PetscRealPart(coord_x),PetscRealPart(coord_y));CHKERRQ(ierr);
            for (d = 0; d < n_dofs; d++) {
                field_d = _fields[n_dofs*((i-si)+(j-sj)*(nx))+d];
                ierr    = PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e ",PetscRealPart(field_d));CHKERRQ(ierr);
            }
            ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"\n");CHKERRQ(ierr);
        }
    }
    ierr = VecRestoreArray(local_fields,&_fields);CHKERRQ(ierr);
    ierr = VecDestroy(&local_fields);CHKERRQ(ierr);

    ierr = DMDAVecRestoreArray(cda,coords,&_coords);CHKERRQ(ierr);

    ierr = PetscFClose(PETSC_COMM_SELF,fp);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

static void FormStressOperatorQ1(PetscScalar Ke[],PetscScalar coords[],PetscScalar E[],PetscScalar nu[])
{
    PetscInt    ngp;
    PetscScalar gp_xi[GAUSS_POINTS][2];
    PetscScalar gp_weight[GAUSS_POINTS];
    PetscInt    p,i,j,k,l;
    PetscScalar GNi_p[NSD][NODES_PER_EL],GNx_p[NSD][NODES_PER_EL];
    PetscScalar J_p;
    PetscScalar B[3][U_DOFS*NODES_PER_EL];
    PetscScalar prop_E,prop_nu,factor,constit_D[3][3];

    /* define quadrature rule */
    ConstructGaussQuadrature(&ngp,gp_xi,gp_weight);

    /* evaluate integral */
    for (p = 0; p < ngp; p++) {
        ConstructQ12D_GNi(gp_xi[p],GNi_p);
        ConstructQ12D_GNx(GNi_p,GNx_p,coords,&J_p);

        for (i = 0; i < NODES_PER_EL; i++) {
            PetscScalar d_dx_i = GNx_p[0][i];
            PetscScalar d_dy_i = GNx_p[1][i];

            B[0][2*i] = d_dx_i;  B[0][2*i+1] = 0.0;
            B[1][2*i] = 0.0;     B[1][2*i+1] = d_dy_i;
            B[2][2*i] = d_dy_i;  B[2][2*i+1] = d_dx_i;
        }

        /* form D for the quadrature point */
        prop_E          = E[p];
        prop_nu         = nu[p];
        factor          = prop_E / ((1.0+prop_nu)*(1.0-2.0*prop_nu));
        constit_D[0][0] = 1.0-prop_nu;  constit_D[0][1] = prop_nu;      constit_D[0][2] = 0.0;
        constit_D[1][0] = prop_nu;      constit_D[1][1] = 1.0-prop_nu;  constit_D[1][2] = 0.0;
        constit_D[2][0] = 0.0;          constit_D[2][1] = 0.0;          constit_D[2][2] = 0.5*(1.0-2.0*prop_nu);
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++) {
                constit_D[i][j] = factor * constit_D[i][j] * gp_weight[p] * J_p;
            }
        }

        /* form Bt tildeD B */
        /*
         Ke_ij = Bt_ik . D_kl . B_lj
         = B_ki . D_kl . B_lj
         */
        for (i = 0; i < 8; i++) {
            for (j = 0; j < 8; j++) {
                for (k = 0; k < 3; k++) {
                    for (l = 0; l < 3; l++) {
                        Ke[8*i+j] = Ke[8*i+j] + B[k][i] * constit_D[k][l] * B[l][j];
                    }
                }
            }
        }

    } /* end quadrature */
}

static void FormMomentumRhsQ1(PetscScalar Fe[],PetscScalar coords[],PetscScalar fx[],PetscScalar fy[])
{
    PetscInt    ngp;
    PetscScalar gp_xi[GAUSS_POINTS][2];
    PetscScalar gp_weight[GAUSS_POINTS];
    PetscInt    p,i;
    PetscScalar Ni_p[NODES_PER_EL];
    PetscScalar GNi_p[NSD][NODES_PER_EL],GNx_p[NSD][NODES_PER_EL];
    PetscScalar J_p,fac;

    /* define quadrature rule */
    ConstructGaussQuadrature(&ngp,gp_xi,gp_weight);

    /* evaluate integral */
    for (p = 0; p < ngp; p++) {
        ConstructQ12D_Ni(gp_xi[p],Ni_p);
        ConstructQ12D_GNi(gp_xi[p],GNi_p);
        ConstructQ12D_GNx(GNi_p,GNx_p,coords,&J_p);
        fac = gp_weight[p]*J_p;

        for (i = 0; i < NODES_PER_EL; i++) {
            Fe[NSD*i]   += fac*Ni_p[i]*fx[p];
            Fe[NSD*i+1] += fac*Ni_p[i]*fy[p];
        }
    }
}

/*
 i,j are the element indices
 The unknown is a vector quantity.
 The s[].c is used to indicate the degree of freedom.
 */
static PetscErrorCode DMDAGetElementEqnums_u(MatStencil s_u[],PetscInt i,PetscInt j)
{
    PetscFunctionBeginUser;
    /* displacement */
    /* node 0 */
    s_u[0].i = i;s_u[0].j = j;s_u[0].c = 0;          /* Ux0 */
    s_u[1].i = i;s_u[1].j = j;s_u[1].c = 1;          /* Uy0 */

    /* node 1 */
    s_u[2].i = i;s_u[2].j = j+1;s_u[2].c = 0;        /* Ux1 */
    s_u[3].i = i;s_u[3].j = j+1;s_u[3].c = 1;        /* Uy1 */

    /* node 2 */
    s_u[4].i = i+1;s_u[4].j = j+1;s_u[4].c = 0;      /* Ux2 */
    s_u[5].i = i+1;s_u[5].j = j+1;s_u[5].c = 1;      /* Uy2 */

    /* node 3 */
    s_u[6].i = i+1;s_u[6].j = j;s_u[6].c = 0;        /* Ux3 */
    s_u[7].i = i+1;s_u[7].j = j;s_u[7].c = 1;        /* Uy3 */
    PetscFunctionReturn(0);
}

static PetscErrorCode GetElementCoords(DMDACoor2d **_coords,PetscInt ei,PetscInt ej,PetscScalar el_coords[])
{
    PetscFunctionBeginUser;
    /* get coords for the element */
    el_coords[NSD*0+0] = _coords[ej][ei].x;      el_coords[NSD*0+1] = _coords[ej][ei].y;
    el_coords[NSD*1+0] = _coords[ej+1][ei].x;    el_coords[NSD*1+1] = _coords[ej+1][ei].y;
    el_coords[NSD*2+0] = _coords[ej+1][ei+1].x;  el_coords[NSD*2+1] = _coords[ej+1][ei+1].y;
    el_coords[NSD*3+0] = _coords[ej][ei+1].x;    el_coords[NSD*3+1] = _coords[ej][ei+1].y;
    PetscFunctionReturn(0);
}

static PetscErrorCode AssembleA_Elasticity(Mat A,DM elas_da,DM properties_da,Vec properties)
{
    DM                     cda;
    Vec                    coords;
    DMDACoor2d             **_coords;
    MatStencil             u_eqn[NODES_PER_EL*U_DOFS]; /* 2 degrees of freedom */
    PetscInt               sex,sey,mx,my;
    PetscInt               ei,ej;
    PetscScalar            Ae[NODES_PER_EL*U_DOFS*NODES_PER_EL*U_DOFS];
    PetscScalar            el_coords[NODES_PER_EL*NSD];
    Vec                    local_properties;
    GaussPointCoefficients **props;
    PetscScalar            *prop_E,*prop_nu;
    PetscErrorCode         ierr;

    PetscFunctionBeginUser;
    /* setup for coords */
    ierr = DMGetCoordinateDM(elas_da,&cda);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(elas_da,&coords);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(cda,coords,&_coords);CHKERRQ(ierr);

    /* setup for coefficients */
    ierr = DMCreateLocalVector(properties_da,&local_properties);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(properties_da,properties,INSERT_VALUES,local_properties);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(properties_da,properties,INSERT_VALUES,local_properties);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(properties_da,local_properties,&props);CHKERRQ(ierr);

    ierr = DMDAGetElementsCorners(elas_da,&sex,&sey,0);CHKERRQ(ierr);
    ierr = DMDAGetElementsSizes(elas_da,&mx,&my,0);CHKERRQ(ierr);
    for (ej = sey; ej < sey+my; ej++) {
        for (ei = sex; ei < sex+mx; ei++) {
            /* get coords for the element */
            GetElementCoords(_coords,ei,ej,el_coords);

            /* get coefficients for the element */
            prop_E  = props[ej][ei].E;
            prop_nu = props[ej][ei].nu;

            /* initialise element stiffness matrix */
            ierr = PetscMemzero(Ae,sizeof(Ae));CHKERRQ(ierr);

            /* form element stiffness matrix */
            FormStressOperatorQ1(Ae,el_coords,prop_E,prop_nu);

            /* insert element matrix into global matrix */
            ierr = DMDAGetElementEqnums_u(u_eqn,ei,ej);CHKERRQ(ierr);
            ierr = MatSetValuesStencil(A,NODES_PER_EL*U_DOFS,u_eqn,NODES_PER_EL*U_DOFS,u_eqn,Ae,ADD_VALUES);CHKERRQ(ierr);
        }
    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    ierr = DMDAVecRestoreArray(cda,coords,&_coords);CHKERRQ(ierr);

    ierr = DMDAVecRestoreArray(properties_da,local_properties,&props);CHKERRQ(ierr);
    ierr = VecDestroy(&local_properties);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

static PetscErrorCode DMDASetValuesLocalStencil_ADD_VALUES(ElasticityDOF **fields_F,MatStencil u_eqn[],PetscScalar Fe_u[])
{
    PetscInt n;

    PetscFunctionBeginUser;
    for (n = 0; n < 4; n++) {
        fields_F[u_eqn[2*n].j][u_eqn[2*n].i].ux_dof     = fields_F[u_eqn[2*n].j][u_eqn[2*n].i].ux_dof+Fe_u[2*n];
        fields_F[u_eqn[2*n+1].j][u_eqn[2*n+1].i].uy_dof = fields_F[u_eqn[2*n+1].j][u_eqn[2*n+1].i].uy_dof+Fe_u[2*n+1];
    }
    PetscFunctionReturn(0);
}

static PetscErrorCode AssembleF_Elasticity(Vec F,DM elas_da,DM properties_da,Vec properties)
{
    DM                     cda;
    Vec                    coords;
    DMDACoor2d             **_coords;
    MatStencil             u_eqn[NODES_PER_EL*U_DOFS]; /* 2 degrees of freedom */
    PetscInt               sex,sey,mx,my;
    PetscInt               ei,ej;
    PetscScalar            Fe[NODES_PER_EL*U_DOFS];
    PetscScalar            el_coords[NODES_PER_EL*NSD];
    Vec                    local_properties;
    GaussPointCoefficients **props;
    PetscScalar            *prop_fx,*prop_fy;
    Vec                    local_F;
    ElasticityDOF          **ff;
    PetscErrorCode         ierr;

    PetscFunctionBeginUser;
    /* setup for coords */
    ierr = DMGetCoordinateDM(elas_da,&cda);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(elas_da,&coords);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(cda,coords,&_coords);CHKERRQ(ierr);

    /* setup for coefficients */
    ierr = DMGetLocalVector(properties_da,&local_properties);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(properties_da,properties,INSERT_VALUES,local_properties);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(properties_da,properties,INSERT_VALUES,local_properties);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(properties_da,local_properties,&props);CHKERRQ(ierr);

    /* get acces to the vector */
    ierr = DMGetLocalVector(elas_da,&local_F);CHKERRQ(ierr);
    ierr = VecZeroEntries(local_F);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(elas_da,local_F,&ff);CHKERRQ(ierr);

    ierr = DMDAGetElementsCorners(elas_da,&sex,&sey,0);CHKERRQ(ierr);
    ierr = DMDAGetElementsSizes(elas_da,&mx,&my,0);CHKERRQ(ierr);
    for (ej = sey; ej < sey+my; ej++) {
        for (ei = sex; ei < sex+mx; ei++) {
            /* get coords for the element */
            GetElementCoords(_coords,ei,ej,el_coords);

            /* get coefficients for the element */
            prop_fx = props[ej][ei].fx;
            prop_fy = props[ej][ei].fy;

            /* initialise element stiffness matrix */
            ierr = PetscMemzero(Fe,sizeof(Fe));CHKERRQ(ierr);

            /* form element stiffness matrix */
            FormMomentumRhsQ1(Fe,el_coords,prop_fx,prop_fy);

            /* insert element matrix into global matrix */
            ierr = DMDAGetElementEqnums_u(u_eqn,ei,ej);CHKERRQ(ierr);

            ierr = DMDASetValuesLocalStencil_ADD_VALUES(ff,u_eqn,Fe);CHKERRQ(ierr);
        }
    }

    ierr = DMDAVecRestoreArray(elas_da,local_F,&ff);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(elas_da,local_F,ADD_VALUES,F);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(elas_da,local_F,ADD_VALUES,F);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(elas_da,&local_F);CHKERRQ(ierr);

    ierr = DMDAVecRestoreArray(cda,coords,&_coords);CHKERRQ(ierr);

    ierr = DMDAVecRestoreArray(properties_da,local_properties,&props);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(properties_da,&local_properties);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

static PetscErrorCode solve_elasticity_2d(PetscInt mx, PetscInt my)
{
    DM                     elas_da,da_prop;
    PetscInt               u_dof,dof,stencil_width;
    Mat                    A;
    PetscInt               mxl,myl;
    DM                     prop_cda,vel_cda;
    Vec                    prop_coords,vel_coords;
    PetscInt               si,sj,nx,ny,i,j,p;
    Vec                    f,X;
    PetscInt               prop_dof,prop_stencil_width;
    Vec                    properties,l_properties;
    MatNullSpace           matnull;
    PetscReal              dx,dy;
    PetscInt               M,N;
    DMDACoor2d             **_prop_coords,**_vel_coords;
    GaussPointCoefficients **element_props;
    KSP                    ksp_E;
    PetscInt               coefficient_structure = 0;
    PetscInt               cpu_x,cpu_y,*lx = nullptr,*ly = nullptr;
    PetscBool              use_gp_coords = PETSC_FALSE;
    PetscBool              use_nonsymbc  = PETSC_FALSE;
    PetscBool              no_view       = PETSC_FALSE;
    PetscBool              flg;
    PetscErrorCode         ierr;

    PetscFunctionBeginUser;
    /* Generate the da for velocity and pressure */
    /*
     We use Q1 elements for the temperature.
     FEM has a 9-point stencil (BOX) or connectivity pattern
     Num nodes in each direction is mx+1, my+1
     */
    u_dof         = U_DOFS; /* Vx, Vy - velocities */
    dof           = u_dof;
    stencil_width = 1;
    ierr          = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, mx+1, my+1,
                                  PETSC_DECIDE, PETSC_DECIDE, dof, stencil_width, nullptr, nullptr, &elas_da);CHKERRQ(ierr);

    ierr = DMSetMatType(elas_da,MATAIJ);CHKERRQ(ierr);
    ierr = DMSetFromOptions(elas_da);CHKERRQ(ierr);
    ierr = DMSetUp(elas_da);CHKERRQ(ierr);

    ierr = DMDASetFieldName(elas_da,0,"Ux");CHKERRQ(ierr);
    ierr = DMDASetFieldName(elas_da,1,"Uy");CHKERRQ(ierr);

    /* unit box [0,1] x [0,1] */
    ierr = DMDASetUniformCoordinates(elas_da,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);

    /* Generate element properties, we will assume all material properties are constant over the element */
    /* local number of elements */
    ierr = DMDAGetElementsSizes(elas_da,&mxl,&myl,nullptr);CHKERRQ(ierr);

    /* !!! IN PARALLEL WE MUST MAKE SURE THE TWO DMDA's ALIGN !!! */
    ierr = DMDAGetInfo(elas_da,0,0,0,0,&cpu_x,&cpu_y,0,0,0,0,0,0,0);CHKERRQ(ierr);
    ierr = DMDAGetElementOwnershipRanges2d(elas_da,&lx,&ly);CHKERRQ(ierr);

    prop_dof           = (PetscInt)(sizeof(GaussPointCoefficients)/sizeof(PetscScalar)); /* gauss point setup */
    prop_stencil_width = 0;
    ierr               = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,mx,my,cpu_x,cpu_y,prop_dof,prop_stencil_width,lx,ly,&da_prop);CHKERRQ(ierr);
    ierr = DMSetFromOptions(da_prop);CHKERRQ(ierr);
    ierr = DMSetUp(da_prop);CHKERRQ(ierr);

    ierr = PetscFree(lx);CHKERRQ(ierr);
    ierr = PetscFree(ly);CHKERRQ(ierr);

    /* define centroid positions */
    ierr = DMDAGetInfo(da_prop,0,&M,&N,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
    dx   = 1.0/((PetscReal)(M));
    dy   = 1.0/((PetscReal)(N));

    ierr = DMDASetUniformCoordinates(da_prop,0.0+0.5*dx,1.0-0.5*dx,0.0+0.5*dy,1.0-0.5*dy,0.0,1.0);CHKERRQ(ierr);

    ierr = DMCreateGlobalVector(da_prop,&properties);CHKERRQ(ierr);
    ierr = DMCreateLocalVector(da_prop,&l_properties);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da_prop,l_properties,&element_props);CHKERRQ(ierr);

    ierr = DMGetCoordinateDM(da_prop,&prop_cda);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(da_prop,&prop_coords);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(prop_cda,prop_coords,&_prop_coords);CHKERRQ(ierr);

    ierr = DMDAGetGhostCorners(prop_cda,&si,&sj,0,&nx,&ny,0);CHKERRQ(ierr);

    ierr = DMGetCoordinateDM(elas_da,&vel_cda);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(elas_da,&vel_coords);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(vel_cda,vel_coords,&_vel_coords);CHKERRQ(ierr);

    /* interpolate the coordinates */
    for (j = sj; j < sj+ny; j++) {
        for (i = si; i < si+nx; i++) {
            PetscInt    ngp;
            PetscScalar gp_xi[GAUSS_POINTS][2],gp_weight[GAUSS_POINTS];
            PetscScalar el_coords[8];

            ierr = GetElementCoords(_vel_coords,i,j,el_coords);CHKERRQ(ierr);
            ConstructGaussQuadrature(&ngp,gp_xi,gp_weight);

            for (p = 0; p < GAUSS_POINTS; p++) {
                PetscScalar gp_x,gp_y;
                PetscInt    n;
                PetscScalar xi_p[2],Ni_p[4];

                xi_p[0] = gp_xi[p][0];
                xi_p[1] = gp_xi[p][1];
                ConstructQ12D_Ni(xi_p,Ni_p);

                gp_x = 0.0;
                gp_y = 0.0;
                for (n = 0; n < NODES_PER_EL; n++) {
                    gp_x = gp_x+Ni_p[n]*el_coords[2*n];
                    gp_y = gp_y+Ni_p[n]*el_coords[2*n+1];
                }
                element_props[j][i].gp_coords[2*p]   = gp_x;
                element_props[j][i].gp_coords[2*p+1] = gp_y;
            }
        }
    }

    for (j = sj; j < sj+ny; j++) {
        for (i = si; i < si + nx; i++) {
            PetscScalar centroid_x = _prop_coords[j][i].x; /* centroids of cell */
            PetscScalar centroid_y = _prop_coords[j][i].y;
            PETSC_UNUSED PetscScalar coord_x, coord_y;

            PetscScalar opts_E,opts_nu;

            opts_E  = 1.0;
            opts_nu = 0.33;
            ierr    = PetscOptionsGetScalar(NULL,NULL,"-iso_E",&opts_E,&flg);CHKERRQ(ierr);
            ierr    = PetscOptionsGetScalar(NULL,NULL,"-iso_nu",&opts_nu,&flg);CHKERRQ(ierr);

            for (p = 0; p < GAUSS_POINTS; p++) {
                element_props[j][i].E[p]  = opts_E;
                element_props[j][i].nu[p] = opts_nu;

                element_props[j][i].fx[p] = 0.0;
                element_props[j][i].fy[p] = 0.0;
            }
        }
    }

    ierr = DMDAVecRestoreArray(prop_cda,prop_coords,&_prop_coords);CHKERRQ(ierr);

    ierr = DMDAVecRestoreArray(vel_cda,vel_coords,&_vel_coords);CHKERRQ(ierr);

    ierr = DMDAVecRestoreArray(da_prop,l_properties,&element_props);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(da_prop,l_properties,ADD_VALUES,properties);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(da_prop,l_properties,ADD_VALUES,properties);CHKERRQ(ierr);

    /* Generate a matrix with the correct non-zero pattern of type AIJ. This will work in parallel and serial */
    ierr = DMCreateMatrix(elas_da,&A);CHKERRQ(ierr);
    ierr = DMGetCoordinates(elas_da,&vel_coords);CHKERRQ(ierr);
    ierr = MatNullSpaceCreateRigidBody(vel_coords,&matnull);CHKERRQ(ierr);
    ierr = MatSetNearNullSpace(A,matnull);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&matnull);CHKERRQ(ierr);
    ierr = MatCreateVecs(A,&f,&X);CHKERRQ(ierr);

    /* assemble A11 */
    ierr = MatZeroEntries(A);CHKERRQ(ierr);
    ierr = VecZeroEntries(f);CHKERRQ(ierr);

    ierr = AssembleA_Elasticity(A,elas_da,da_prop,properties);CHKERRQ(ierr);
    /* build force vector */
    ierr = AssembleF_Elasticity(f,elas_da,da_prop,properties);CHKERRQ(ierr);

    ierr = KSPCreate(PETSC_COMM_WORLD,&ksp_E);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(ksp_E,"elas_");CHKERRQ(ierr);  /* elasticity */

    /* solve */
    ierr = DMDABCApplyCompression(elas_da,A,f);CHKERRQ(ierr);

    ierr = KSPSetOperators(ksp_E,A,A);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp_E);CHKERRQ(ierr);

    ierr = KSPSolve(ksp_E,f,X);CHKERRQ(ierr);

    ierr = DMDAViewGnuplot2d(elas_da,X,"Displacement solution for elasticity eqn.","X");CHKERRQ(ierr);

    ierr = KSPDestroy(&ksp_E);CHKERRQ(ierr);

    ierr = VecDestroy(&X);CHKERRQ(ierr);
    ierr = VecDestroy(&f);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);

    ierr = DMDestroy(&elas_da);CHKERRQ(ierr);
    ierr = DMDestroy(&da_prop);CHKERRQ(ierr);

    ierr = VecDestroy(&properties);CHKERRQ(ierr);
    ierr = VecDestroy(&l_properties);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


int main(int argc,char **args)
{
    PetscErrorCode  ierr;
    PetscInt        nel;

    ierr = PetscInitialize(&argc,&args,(char*)0,help); if (ierr) return ierr;
    nel  = 10;
    ierr = PetscOptionsGetInt(NULL,NULL,"-nel",&nel,NULL);CHKERRQ(ierr);
    ierr = solve_elasticity_2d(nel,nel);CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;

}


/* -------------------------- helpers for boundary conditions -------------------------------- */

static PetscErrorCode BCApply_EAST(DM da,PetscInt d_idx,PetscScalar bc_val,Mat A,Vec b)
{
    DM                     cda;
    Vec                    coords;
    PetscInt               si,sj,nx,ny,i,j;
    PetscInt               M,N;
    DMDACoor2d             **_coords;
    const PetscInt         *g_idx;
    PetscInt               *bc_global_ids;
    PetscScalar            *bc_vals;
    PetscInt               nbcs;
    PetscInt               n_dofs;
    PetscErrorCode         ierr;
    ISLocalToGlobalMapping ltogm;

    PetscFunctionBeginUser;
    /* enforce bc's */
    ierr = DMGetLocalToGlobalMapping(da,&ltogm);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetIndices(ltogm,&g_idx);CHKERRQ(ierr);

    ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(da,&coords);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(cda,coords,&_coords);CHKERRQ(ierr);
    ierr = DMDAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0);CHKERRQ(ierr);
    ierr = DMDAGetInfo(da,0,&M,&N,0,0,0,0,&n_dofs,0,0,0,0,0);CHKERRQ(ierr);

    /* --- */

    ierr = PetscMalloc1(ny*n_dofs,&bc_global_ids);CHKERRQ(ierr);
    ierr = PetscMalloc1(ny*n_dofs,&bc_vals);CHKERRQ(ierr);

    /* init the entries to -1 so VecSetValues will ignore them */
    for (i = 0; i < ny*n_dofs; i++) bc_global_ids[i] = -1;

    i = nx-1;
    for (j = 0; j < ny; j++) {
        PetscInt                 local_id;
        PETSC_UNUSED PetscScalar coordx,coordy;

        local_id = i+j*nx;

        bc_global_ids[j] = g_idx[n_dofs*local_id+d_idx];

        coordx = _coords[j+sj][i+si].x;
        coordy = _coords[j+sj][i+si].y;

        bc_vals[j] =  bc_val;
    }
    ierr = ISLocalToGlobalMappingRestoreIndices(ltogm,&g_idx);CHKERRQ(ierr);
    nbcs = 0;
    if ((si+nx) == (M)) nbcs = ny;

    if (b) {
        ierr = VecSetValues(b,nbcs,bc_global_ids,bc_vals,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
    }
    if (A) {
        ierr = MatZeroRows(A,nbcs,bc_global_ids,1.0,0,0);CHKERRQ(ierr);
    }

    ierr = PetscFree(bc_vals);CHKERRQ(ierr);
    ierr = PetscFree(bc_global_ids);CHKERRQ(ierr);

    ierr = DMDAVecRestoreArray(cda,coords,&_coords);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

static PetscErrorCode BCApply_WEST(DM da,PetscInt d_idx,PetscScalar bc_val,Mat A,Vec b)
{
    DM                     cda;
    Vec                    coords;
    PetscInt               si,sj,nx,ny,i,j;
    PetscInt               M,N;
    DMDACoor2d             **_coords;
    const PetscInt         *g_idx;
    PetscInt               *bc_global_ids;
    PetscScalar            *bc_vals;
    PetscInt               nbcs;
    PetscInt               n_dofs;
    PetscErrorCode         ierr;
    ISLocalToGlobalMapping ltogm;

    PetscFunctionBeginUser;
    /* enforce bc's */
    ierr = DMGetLocalToGlobalMapping(da,&ltogm);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetIndices(ltogm,&g_idx);CHKERRQ(ierr);

    ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(da,&coords);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(cda,coords,&_coords);CHKERRQ(ierr);
    ierr = DMDAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0);CHKERRQ(ierr);
    ierr = DMDAGetInfo(da,0,&M,&N,0,0,0,0,&n_dofs,0,0,0,0,0);CHKERRQ(ierr);

    /* --- */

    ierr = PetscMalloc1(ny*n_dofs,&bc_global_ids);CHKERRQ(ierr);
    ierr = PetscMalloc1(ny*n_dofs,&bc_vals);CHKERRQ(ierr);

    /* init the entries to -1 so VecSetValues will ignore them */
    for (i = 0; i < ny*n_dofs; i++) bc_global_ids[i] = -1;

    i = 0;
    for (j = 0; j < ny; j++) {
        PetscInt                 local_id;
        PETSC_UNUSED PetscScalar coordx,coordy;

        local_id = i+j*nx;

        bc_global_ids[j] = g_idx[n_dofs*local_id+d_idx];

        coordx = _coords[j+sj][i+si].x;
        coordy = _coords[j+sj][i+si].y;

        bc_vals[j] =  bc_val;
    }
    ierr = ISLocalToGlobalMappingRestoreIndices(ltogm,&g_idx);CHKERRQ(ierr);
    nbcs = 0;
    if (si == 0) nbcs = ny;

    if (b) {
        ierr = VecSetValues(b,nbcs,bc_global_ids,bc_vals,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
    }
    if (A) {
        ierr = MatZeroRows(A,nbcs,bc_global_ids,1.0,0,0);CHKERRQ(ierr);
    }

    ierr = PetscFree(bc_vals);CHKERRQ(ierr);
    ierr = PetscFree(bc_global_ids);CHKERRQ(ierr);

    ierr = DMDAVecRestoreArray(cda,coords,&_coords);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

static PetscErrorCode DMDABCApplyCompression(DM elas_da,Mat A,Vec f)
{
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    ierr = BCApply_EAST(elas_da,0,-1.0,A,f);CHKERRQ(ierr);
    ierr = BCApply_EAST(elas_da,1, 0.0,A,f);CHKERRQ(ierr);
    ierr = BCApply_WEST(elas_da,0,1.0,A,f);CHKERRQ(ierr);
    ierr = BCApply_WEST(elas_da,1,0.0,A,f);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}