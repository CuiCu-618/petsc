/*
    Concepts: KSP^solving a system of linear equations
    Concepts: KSP^Laplacian, 3d
    Processors: n
 */


static char help[] = "Solves 3D Poisson equation.\\n\\n";

#include <iostream>
#include <iomanip>
#include <petscksp.h>
#include <petscdmda.h>
#include <petscdm.h>

static PetscErrorCode DMDABCApplyCompression(DM,Mat,Vec);

#define NSD            3 /* number of spatial dimensions */
#define NODES_PER_EL   8 /* nodes per element */
#define U_DOF          1
#define GAUSS_POINTS   8

/* FEM routines */
/*
 Element: Local basis function ordering
 1-----2
 |     |
 |     |
 0-----3
 */
static void ShapeFunctionQ13D_Evaluate(PetscScalar _xi[],PetscScalar Ni[])
{
    PetscReal xi   = PetscRealPart(_xi[0]);
    PetscReal eta  = PetscRealPart(_xi[1]);
    PetscReal zeta = PetscRealPart(_xi[2]);

    Ni[0] = 0.125 * (1.0 - xi) * (1.0 - eta) * (1.0 - zeta);
    Ni[1] = 0.125 * (1.0 - xi) * (1.0 + eta) * (1.0 - zeta);
    Ni[2] = 0.125 * (1.0 + xi) * (1.0 + eta) * (1.0 - zeta);
    Ni[3] = 0.125 * (1.0 + xi) * (1.0 - eta) * (1.0 - zeta);

    Ni[4] = 0.125 * (1.0 - xi) * (1.0 - eta) * (1.0 + zeta);
    Ni[5] = 0.125 * (1.0 - xi) * (1.0 + eta) * (1.0 + zeta);
    Ni[6] = 0.125 * (1.0 + xi) * (1.0 + eta) * (1.0 + zeta);
    Ni[7] = 0.125 * (1.0 + xi) * (1.0 - eta) * (1.0 + zeta);
}

static void ShapeFunctionQ13D_Evaluate_dxi(PetscScalar _xi[],PetscScalar GNi[][NODES_PER_EL])
{
    PetscReal xi   = PetscRealPart(_xi[0]);
    PetscReal eta  = PetscRealPart(_xi[1]);
    PetscReal zeta = PetscRealPart(_xi[2]);
    /* xi deriv */
    GNi[0][0] = -0.125 * (1.0 - eta) * (1.0 - zeta);
    GNi[0][1] = -0.125 * (1.0 + eta) * (1.0 - zeta);
    GNi[0][2] =  0.125 * (1.0 + eta) * (1.0 - zeta);
    GNi[0][3] =  0.125 * (1.0 - eta) * (1.0 - zeta);

    GNi[0][4] = -0.125 * (1.0 - eta) * (1.0 + zeta);
    GNi[0][5] = -0.125 * (1.0 + eta) * (1.0 + zeta);
    GNi[0][6] =  0.125 * (1.0 + eta) * (1.0 + zeta);
    GNi[0][7] =  0.125 * (1.0 - eta) * (1.0 + zeta);
    /* eta deriv */
    GNi[1][0] = -0.125 * (1.0 - xi) * (1.0 - zeta);
    GNi[1][1] =  0.125 * (1.0 - xi) * (1.0 - zeta);
    GNi[1][2] =  0.125 * (1.0 + xi) * (1.0 - zeta);
    GNi[1][3] = -0.125 * (1.0 + xi) * (1.0 - zeta);

    GNi[1][4] = -0.125 * (1.0 - xi) * (1.0 + zeta);
    GNi[1][5] =  0.125 * (1.0 - xi) * (1.0 + zeta);
    GNi[1][6] =  0.125 * (1.0 + xi) * (1.0 + zeta);
    GNi[1][7] = -0.125 * (1.0 + xi) * (1.0 + zeta);
    /* zeta deriv */
    GNi[2][0] = -0.125 * (1.0 - xi) * (1.0 - eta);
    GNi[2][1] = -0.125 * (1.0 - xi) * (1.0 + eta);
    GNi[2][2] = -0.125 * (1.0 + xi) * (1.0 + eta);
    GNi[2][3] = -0.125 * (1.0 + xi) * (1.0 - eta);

    GNi[2][4] = 0.125 * (1.0 - xi) * (1.0 - eta);
    GNi[2][5] = 0.125 * (1.0 - xi) * (1.0 + eta);
    GNi[2][6] = 0.125 * (1.0 + xi) * (1.0 + eta);
    GNi[2][7] = 0.125 * (1.0 + xi) * (1.0 - eta);
}


static void matrix_inverse_3x3(PetscScalar A[3][3],PetscScalar B[3][3])
{
    PetscScalar t4, t6, t8, t10, t12, t14, t17;

    t4  = A[2][0] * A[0][1];
    t6  = A[2][0] * A[0][2];
    t8  = A[1][0] * A[0][1];
    t10 = A[1][0] * A[0][2];
    t12 = A[0][0] * A[1][1];
    t14 = A[0][0] * A[1][2];
    t17 = 0.1e1 / (t4 * A[1][2] - t6 * A[1][1] - t8 * A[2][2] + t10 * A[2][1] + t12 * A[2][2] - t14 * A[2][1]);

    B[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * t17;
    B[0][1] = -(A[0][1] * A[2][2] - A[0][2] * A[2][1]) * t17;
    B[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * t17;
    B[1][0] = -(-A[2][0] * A[1][2] + A[1][0] * A[2][2]) * t17;
    B[1][1] = (-t6 + A[0][0] * A[2][2]) * t17;
    B[1][2] = -(-t10 + t14) * t17;
    B[2][0] = (-A[2][0] * A[1][1] + A[1][0] * A[2][1]) * t17;
    B[2][1] = -(-t4 + A[0][0] * A[2][1]) * t17;
    B[2][2] = (-t8 + t12) * t17;
}

static void ShapeFunctionQ13D_Evaluate_dx(PetscScalar GNi[][NODES_PER_EL],PetscScalar GNx[][NODES_PER_EL],PetscScalar coords[],PetscScalar *det_J)
{
    PetscScalar J00,J01,J02,J10,J11,J12,J20,J21,J22;
    PetscInt    n;
    PetscScalar iJ[3][3],JJ[3][3];

    J00 = J01 = J02 = 0.0;
    J10 = J11 = J12 = 0.0;
    J20 = J21 = J22 = 0.0;
    for (n=0; n<NODES_PER_EL; n++) {
        PetscScalar cx = coords[NSD*n + 0];
        PetscScalar cy = coords[NSD*n + 1];
        PetscScalar cz = coords[NSD*n + 2];

        /* J_ij = d(x_j) / d(xi_i) */ /* J_ij = \sum _I GNi[j][I} * x_i */
        J00 = J00 + GNi[0][n] * cx;   /* J_xx */
        J01 = J01 + GNi[0][n] * cy;   /* J_xy = dx/deta */
        J02 = J02 + GNi[0][n] * cz;   /* J_xz = dx/dzeta */

        J10 = J10 + GNi[1][n] * cx;   /* J_yx = dy/dxi */
        J11 = J11 + GNi[1][n] * cy;   /* J_yy */
        J12 = J12 + GNi[1][n] * cz;   /* J_yz */

        J20 = J20 + GNi[2][n] * cx;   /* J_zx */
        J21 = J21 + GNi[2][n] * cy;   /* J_zy */
        J22 = J22 + GNi[2][n] * cz;   /* J_zz */
    }

    JJ[0][0] = J00;      JJ[0][1] = J01;      JJ[0][2] = J02;
    JJ[1][0] = J10;      JJ[1][1] = J11;      JJ[1][2] = J12;
    JJ[2][0] = J20;      JJ[2][1] = J21;      JJ[2][2] = J22;

    matrix_inverse_3x3(JJ,iJ);

    *det_J = J00*J11*J22 - J00*J12*J21 - J10*J01*J22 + J10*J02*J21 + J20*J01*J12 - J20*J02*J11;

    for (n=0; n<NODES_PER_EL; n++) {
        GNx[0][n] = GNi[0][n]*iJ[0][0] + GNi[1][n]*iJ[0][1] + GNi[2][n]*iJ[0][2];
        GNx[1][n] = GNi[0][n]*iJ[1][0] + GNi[1][n]*iJ[1][1] + GNi[2][n]*iJ[1][2];
        GNx[2][n] = GNi[0][n]*iJ[2][0] + GNi[1][n]*iJ[2][1] + GNi[2][n]*iJ[2][2];
    }
}

static void ConstructGaussQuadrature3D(PetscInt *ngp,PetscScalar gp_xi[][NSD],PetscScalar gp_weight[])
{
    *ngp        = 8;
    gp_xi[0][0] = -0.57735026919; gp_xi[0][1] = -0.57735026919; gp_xi[0][2] = -0.57735026919;
    gp_xi[1][0] = -0.57735026919; gp_xi[1][1] =  0.57735026919; gp_xi[1][2] = -0.57735026919;
    gp_xi[2][0] =  0.57735026919; gp_xi[2][1] =  0.57735026919; gp_xi[2][2] = -0.57735026919;
    gp_xi[3][0] =  0.57735026919; gp_xi[3][1] = -0.57735026919; gp_xi[3][2] = -0.57735026919;

    gp_xi[4][0] = -0.57735026919; gp_xi[4][1] = -0.57735026919; gp_xi[4][2] =  0.57735026919;
    gp_xi[5][0] = -0.57735026919; gp_xi[5][1] =  0.57735026919; gp_xi[5][2] =  0.57735026919;
    gp_xi[6][0] =  0.57735026919; gp_xi[6][1] =  0.57735026919; gp_xi[6][2] =  0.57735026919;
    gp_xi[7][0] =  0.57735026919; gp_xi[7][1] = -0.57735026919; gp_xi[7][2] =  0.57735026919;

    gp_weight[0] = 1.0;
    gp_weight[1] = 1.0;
    gp_weight[2] = 1.0;
    gp_weight[3] = 1.0;

    gp_weight[4] = 1.0;
    gp_weight[5] = 1.0;
    gp_weight[6] = 1.0;
    gp_weight[7] = 1.0;
}

/*
 i,j are the element indices
 The unknown is a vector quantity.
 The s[].c is used to indicate the degree of freedom.
 */
static PetscErrorCode DMDAGetElementEqnums3D_up(MatStencil s_u[],PetscInt i,PetscInt j,PetscInt k)
{
    PetscInt n;

    PetscFunctionBeginUser;
    /* pressure */
    n = 0;
    /* node 0 */
    s_u[n].i = i;   s_u[n].j = j;   s_u[n].k = k; s_u[n].c = 0; n++; /* Ux0 */
    s_u[n].i = i;   s_u[n].j = j+1; s_u[n].k = k; s_u[n].c = 0; n++;
    s_u[n].i = i+1; s_u[n].j = j+1; s_u[n].k = k; s_u[n].c = 0; n++;
    s_u[n].i = i+1; s_u[n].j = j;   s_u[n].k = k; s_u[n].c = 0; n++;

    /* */
    s_u[n].i = i;   s_u[n].j = j;   s_u[n].k = k+1; s_u[n].c = 0; n++; /* Ux4 */
    s_u[n].i = i;   s_u[n].j = j+1; s_u[n].k = k+1; s_u[n].c = 0; n++;
    s_u[n].i = i+1; s_u[n].j = j+1; s_u[n].k = k+1; s_u[n].c = 0; n++;
    s_u[n].i = i+1; s_u[n].j = j;   s_u[n].k = k+1; s_u[n].c = 0; n++;

    return 0;
}

static PetscErrorCode GetElementCoords3D(DMDACoor3d ***coords,PetscInt i,PetscInt j, PetscInt k,PetscScalar el_coord[])
{
    /* get coords for the element */
    el_coord[0] = coords[k][j][i].x;
    el_coord[1] = coords[k][j][i].y;
    el_coord[2] = coords[k][j][i].z;

    el_coord[3] = coords[k][j+1][i].x;
    el_coord[4] = coords[k][j+1][i].y;
    el_coord[5] = coords[k][j+1][i].z;

    el_coord[6] = coords[k][j+1][i+1].x;
    el_coord[7] = coords[k][j+1][i+1].y;
    el_coord[8] = coords[k][j+1][i+1].z;

    el_coord[9]  = coords[k][j][i+1].x;
    el_coord[10] = coords[k][j][i+1].y;
    el_coord[11] = coords[k][j][i+1].z;

    el_coord[12] = coords[k+1][j][i].x;
    el_coord[13] = coords[k+1][j][i].y;
    el_coord[14] = coords[k+1][j][i].z;

    el_coord[15] = coords[k+1][j+1][i].x;
    el_coord[16] = coords[k+1][j+1][i].y;
    el_coord[17] = coords[k+1][j+1][i].z;

    el_coord[18] = coords[k+1][j+1][i+1].x;
    el_coord[19] = coords[k+1][j+1][i+1].y;
    el_coord[20] = coords[k+1][j+1][i+1].z;

    el_coord[21] = coords[k+1][j][i+1].x;
    el_coord[22] = coords[k+1][j][i+1].y;
    el_coord[23] = coords[k+1][j][i+1].z;
    return 0;
}

static PetscErrorCode PoissonDAGetNodalFields3D(PetscScalar ***field,PetscInt i,PetscInt j,PetscInt k,PetscScalar nodal_fields[])
{
    /* get the nodal fields */
    nodal_fields[0] = field[k][j][i];
    nodal_fields[1] = field[k][j+1][i];
    nodal_fields[2] = field[k][j+1][i+1];
    nodal_fields[3] = field[k][j][i+1];

    nodal_fields[4] = field[k+1][j][i];
    nodal_fields[5] = field[k+1][j+1][i];
    nodal_fields[6] = field[k+1][j+1][i+1];
    nodal_fields[7] = field[k+1][j][i+1];

    return 0;
}


static void FormStressOperatorQ13D(PetscScalar Ke[],PetscScalar coords[])
{
    PetscInt        ngp;
    PetscScalar     gp_xi[GAUSS_POINTS][NSD];
    PetscScalar     gp_weight[GAUSS_POINTS];
    PetscInt        p,i,j,k;
    PetscScalar     GNi_p[NSD][NODES_PER_EL],GNx_p[NSD][NODES_PER_EL];
    PetscScalar     J_p;
    const PetscInt  nvdof = U_DOF*NODES_PER_EL;

    /* define quadrature rule */
    ConstructGaussQuadrature3D(&ngp,gp_xi,gp_weight);

    /* evaluate integral */
    for (p = 0; p < ngp; p++){
        ShapeFunctionQ13D_Evaluate_dxi(gp_xi[p],GNi_p);
        ShapeFunctionQ13D_Evaluate_dx(GNi_p,GNx_p,coords,&J_p);

        /* form GNx_p' * GNx_p */
        for (i = 0; i < nvdof; i++){
            for ( j = 0; j < nvdof; j++){
                for ( k = 0; k < 3; k++){
                    Ke[i*nvdof+j] += GNx_p[k][i] * GNx_p[k][j] * gp_weight[p] * J_p;
                }
            }
        }
    }
}

#define _ZERO_ROW_i(A,i) {                   \
    PetscInt    KK;                             \
    PetscScalar tmp = 1.0;                      \
    for (KK=0;KK<8;KK++) A[8*(i)+KK]=0.0;       \
    A[8*(i)+(i)] = tmp;}                        \


static PetscErrorCode AssembleA_Poisson(Mat A,DM da)
{
    DM              cda;
    Vec             coords;
    DMDACoor3d      ***_coords;
    PetscInt        ei,ej,ek,dim,dof;
    PetscInt        n,M,N,P,xs,ys,zs,xm,ym,zm;
    PetscScalar     el_coords[NODES_PER_EL*NSD];
    PetscScalar     Ae[NODES_PER_EL*U_DOF*NODES_PER_EL*U_DOF];
    MatStencil      u_eqn[NODES_PER_EL*U_DOF];


    DMDAGetInfo(da,&dim,&M,&N,&P,0,0,0,&dof,0,0,0,0,0);

    /* setup for coords */
    DMGetCoordinateDM(da,&cda);
    DMGetCoordinatesLocal(da,&coords);
    DMDAVecGetArray(cda,coords,&_coords);
    DMDAGetElementsCorners(da,&xs,&ys,&zs);
    DMDAGetElementsSizes(da,&xm,&ym,&zm);
    for (ek = zs; ek < zs+zm; ek++) {
        for (ej = ys; ej < ys+ym; ej++) {
            for (ei = xs; ei < xs+xm; ei++) {
                /* get coords for the element */
                GetElementCoords3D(_coords,ei,ej,ek,el_coords);

                /* initialise element stiffness matrix */
                PetscMemzero(Ae, sizeof(Ae));

                /* form element stiffness matrix */
                FormStressOperatorQ13D(Ae,el_coords);

                /* insert element matrix into global matrix */
                DMDAGetElementEqnums3D_up(u_eqn,ei,ej,ek);

                for (n=0; n<NODES_PER_EL; n++){
                    if ((u_eqn[n].i == 0) || (u_eqn[n].i == M-1)){
                        _ZERO_ROW_i(Ae,n);
                    }
                    if ((u_eqn[n].j == 0) || (u_eqn[n].j == N-1)){
                        _ZERO_ROW_i(Ae,n);
                    }
                    if ((u_eqn[n].k == 0) || (u_eqn[n].k == P-1)){
                        _ZERO_ROW_i(Ae,n);
                    }
                }

                MatSetValuesStencil(A,NODES_PER_EL*U_DOF,u_eqn,NODES_PER_EL*U_DOF,u_eqn,Ae,ADD_VALUES);
            }
        }
    }

    MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

    DMDAVecRestoreArray(cda,coords,&_coords);
    return 0;
}

static void evaluate_fv(PetscReal pos[],PetscScalar *fv)
{
    PetscReal x,y,z;
    x = pos[0];
    y = pos[1];
    z = pos[2];
    *fv = 3.0*PETSC_PI*PETSC_PI*PetscSinReal(PETSC_PI*x)*PetscSinReal(PETSC_PI*y)*PetscSinReal(PETSC_PI*z);
}

static void evaluate_ExactFunction(PetscReal pos[],PetscScalar *u)
{
    PetscScalar x,y,z;
    x = pos[0];
    y = pos[1];
    z = pos[2];
    *u = PetscSinReal(PETSC_PI*x)*PetscSinReal(PETSC_PI*y)*PetscSinReal(PETSC_PI*z);
}

static PetscErrorCode DMDACreateManufacturedSolution(PetscInt mx,PetscInt my,PetscInt mz,DM *_da,Vec *_X)
{
    DM              da,cda;
    Vec             X;
    PetscScalar     ***ff;
    Vec             coords;
    DMDACoor3d      ***_coords;
    PetscInt        si,sj,sk,ei,ej,ek,i,j,k;

    DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,
                         mx+1,my+1,mz+1,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,nullptr,nullptr,nullptr,&da);
    DMSetFromOptions(da);
    DMSetUp(da);
    DMDASetFieldName(da,0,"analytic_P");

    DMDASetUniformCoordinates(da,0,1,0,1,0,1);

    DMGetCoordinatesLocal(da,&coords);
    DMGetCoordinateDM(da,&cda);
    DMDAVecGetArray(cda,coords,&_coords);

    DMCreateGlobalVector(da,&X);
    DMDAVecGetArray(da,X,&ff);

    DMDAGetCorners(da,&si,&sj,&sk,&ei,&ej,&ek);
    for (k = sk; k < sk+ek; k++) {
        for (j = sj; j < sj + ej; j++) {
            for (i = si; i < si + ei; i++) {
                PetscReal pos[NSD], u;
                pos[0] = PetscRealPart(_coords[k][j][i].x);
                pos[1] = PetscRealPart(_coords[k][j][i].y);
                pos[2] = PetscRealPart(_coords[k][j][i].z);

                evaluate_ExactFunction(pos,&u);
                ff[k][j][i] = u;
            }
        }
    }
    DMDAVecRestoreArray(da,X,&ff);
    DMDAVecRestoreArray(cda,coords,&_coords);

    *_da = da;
    *_X  = X;
    return 0;
}

static PetscErrorCode DMDAIntegrateErrors3D(DM da,Vec X,Vec X_analytic)
{
    DM              cda;
    Vec             coords,X_analytic_local,X_local;
    DMDACoor3d      ***_coords;
    PetscInt        sex,sey,sez,mx,my,mz;
    PetscInt        ei,ej,ek;
    PetscScalar     el_coords[NODES_PER_EL*NSD];
    PetscScalar     ***ff,***ff_analytic;
    PetscScalar     ff_e[NODES_PER_EL],ff_analytic_e[NODES_PER_EL];

    PetscScalar     GNi_p[NSD][NODES_PER_EL],GNx_p[NSD][NODES_PER_EL];
    PetscScalar     Ni_p[NODES_PER_EL];
    PetscInt        ngp;
    PetscScalar     gp_xi[GAUSS_POINTS][NSD];
    PetscScalar     gp_weight[GAUSS_POINTS];
    PetscInt        p,i;
    PetscScalar     J_p,fac;
    PetscScalar     h,u_e_int,u_e_intH,u_e_L2,u_e_H1,u_int,u_intH,u_L2,u_H1,tu_int,tu_intH,tu_L2,tu_H1;
    PetscInt        M;
    PetscReal       xymin[NSD],xymax[NSD];
    PetscErrorCode  ierr;

    /* define quadrature rule */
    ConstructGaussQuadrature3D(&ngp,gp_xi,gp_weight);

    /* setup for coords */
    DMGetCoordinatesLocal(da,&coords);
    DMGetCoordinateDM(da,&cda);
    DMDAVecGetArray(cda,coords,&_coords);

    /* setup for analytic */
    DMCreateLocalVector(da,&X_analytic_local);
    DMGlobalToLocalBegin(da,X_analytic,INSERT_VALUES,X_analytic_local);
    DMGlobalToLocalEnd(da,X_analytic,INSERT_VALUES,X_analytic_local);
    DMDAVecGetArray(da,X_analytic_local,&ff_analytic);

    /* setup for solution */
    DMCreateLocalVector(da,&X_local);
    DMGlobalToLocalBegin(da,X,INSERT_VALUES,X_local);
    DMGlobalToLocalEnd(da,X,INSERT_VALUES,X_local);
    DMDAVecGetArray(da,X_local,&ff);

    DMDAGetInfo(da,0,&M,0,0,0,0,0,0,0,0,0,0,0);
    DMGetBoundingBox(da,xymin,xymax);
    h = (xymax[0]-xymin[0])/((PetscReal)(M-1));

    tu_int = tu_intH = 0.0;
    tu_L2  = tu_H1 = 0.0;

    DMDAGetElementsCorners(da,&sex,&sey,&sez);
    DMDAGetElementsSizes(da,&mx,&my,&mz);
    for (ek = sez; ek < sez+mz; ek++) {
        for (ej = sey; ej < sey + my; ej++) {
            for (ei = sex; ei < sex + mx; ei++) {
                /* get coords for the element */
                GetElementCoords3D(_coords, ei, ej, ek, el_coords);
                PoissonDAGetNodalFields3D(ff,ei,ej,ek,ff_e);
                PoissonDAGetNodalFields3D(ff_analytic,ei,ej,ek,ff_analytic_e);

                /* evaluate integral */
                u_e_L2   = 0.0;
                u_e_H1   = 0.0;
                u_e_int  = 0.0;
                u_e_intH = 0.0;
                for (p = 0; p < ngp; p++) {
                    ShapeFunctionQ13D_Evaluate(gp_xi[p], Ni_p);
                    ShapeFunctionQ13D_Evaluate_dxi(gp_xi[p], GNi_p);
                    ShapeFunctionQ13D_Evaluate_dx(GNi_p, GNx_p, el_coords, &J_p);
                    fac = gp_weight[p] * J_p;

                    for (i = 0; i < NODES_PER_EL; i++){
                        PetscScalar u_error;
                        u_error  = ff_e[i] - ff_analytic_e[i];
                        u_e_L2  += fac*Ni_p[i]*u_error*u_error;

                        u_e_H1  += fac*(GNx_p[0][i]*u_error*GNx_p[0][i]*u_error
                                        +GNx_p[1][i]*u_error*GNx_p[1][i]*u_error
                                        +GNx_p[2][i]*u_error*GNx_p[2][i]*u_error);

                        u_e_int  += fac*Ni_p[i]*ff_analytic_e[i]*ff_analytic_e[i];
                        u_e_intH += fac*(GNx_p[0][i]*ff_analytic_e[i]*GNx_p[0][i]*ff_analytic_e[i]
                                         +GNx_p[1][i]*ff_analytic_e[i]*GNx_p[1][i]*ff_analytic_e[i]
                                         +GNx_p[2][i]*ff_analytic_e[i]*GNx_p[2][i]*ff_analytic_e[i]);
                    }
                }
                tu_L2  += u_e_L2;
                tu_H1  += u_e_H1;

                tu_int  += u_e_int;
                tu_intH += u_e_intH;
            }
        }
    }
    MPI_Allreduce(&tu_L2,&u_L2,1,MPIU_SCALAR,MPIU_SUM,PETSC_COMM_WORLD);
    MPI_Allreduce(&tu_H1,&u_H1,1,MPIU_SCALAR,MPIU_SUM,PETSC_COMM_WORLD);
    MPI_Allreduce(&tu_int,&u_int,1,MPIU_SCALAR,MPIU_SUM,PETSC_COMM_WORLD);
    MPI_Allreduce(&tu_intH,&u_intH,1,MPIU_SCALAR,MPIU_SUM,PETSC_COMM_WORLD);

    u_L2  = PetscSqrtScalar(u_L2);
    u_H1  = PetscSqrtScalar(u_H1);

    u_int  = PetscSqrtScalar(u_int);
    u_intH = PetscSqrtScalar(u_intH);

//    PetscPrintf(PETSC_COMM_WORLD,"%1.4e   %1.4e   %1.4e \n",PetscRealPart(h),
//            PetscRealPart(u_L2)/PetscRealPart(u_int),PetscRealPart(u_H1)/PetscRealPart(u_intH));

    PetscPrintf(PETSC_COMM_WORLD,"%1.4e   %1.4e   %1.4e \n",PetscRealPart(h),PetscRealPart(u_L2),PetscRealPart(u_H1));
    ierr = DMDAVecRestoreArray(cda,coords,&_coords);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da,X_analytic_local,&ff_analytic);CHKERRQ(ierr);
    VecDestroy(&X_analytic_local);
    ierr = DMDAVecRestoreArray(da,X_local,&ff);CHKERRQ(ierr);
    VecDestroy(&X_local);

    return 0;
}

static void FormRHSQ13D(PetscScalar Fe[],PetscScalar coords[])
{
    PetscInt        ngp;
    PetscScalar     gp_xi[GAUSS_POINTS][NSD];
    PetscScalar     gp_weight[GAUSS_POINTS];
    PetscInt        n,p,i;
    PetscScalar     Ni_p[NODES_PER_EL];
    PetscScalar     GNi_p[NSD][NODES_PER_EL],GNx_p[NSD][NODES_PER_EL];
    PetscScalar     J_p,fac;
    PetscScalar     fv;

    /* define quadrature rule */
    ConstructGaussQuadrature3D(&ngp,gp_xi,gp_weight);

    /* evaluate integral */
    for (p = 0; p < ngp; p++) {
        ShapeFunctionQ13D_Evaluate(gp_xi[p],Ni_p);
        ShapeFunctionQ13D_Evaluate_dxi(gp_xi[p],GNi_p);
        ShapeFunctionQ13D_Evaluate_dx(GNi_p,GNx_p,coords,&J_p);
        fac = gp_weight[p]*J_p;

        PetscReal pos[NSD];
        pos[0] = pos[1] = pos[2] = 0.0;
        for (n = 0; n < NODES_PER_EL; n++){
            pos[0] += Ni_p[n] * coords[NSD*n];
            pos[1] += Ni_p[n] * coords[NSD*n+1];
            pos[2] += Ni_p[n] * coords[NSD*n+2];
        }
        evaluate_fv(pos,&fv);
        for (i=0; i<NODES_PER_EL; i++){
            Fe[i] += fac * fv * Ni_p[i];
        }
    }
}

static PetscErrorCode DMDASetValuesLocalStencil3D_ADD_VALUES(PetscScalar ***ff,MatStencil u_eqn[],PetscScalar Fe[])
{
    PetscInt    n,I,J,K;
    for (n = 0; n < NODES_PER_EL; n++){
        I = u_eqn[n].i;
        J = u_eqn[n].j;
        K = u_eqn[n].k;
        ff[K][J][I] += Fe[n];
    }
    return 0;
}

static PetscErrorCode AssembleF_Poisson(Vec F,DM da)
{
    DM              cda;
    Vec             coords;
    DMDACoor3d      ***_coords;
    MatStencil      u_eqn[NODES_PER_EL*U_DOF];
    PetscInt        sex,sey,sez,mx,my,mz;
    PetscInt        ei,ej,ek;
    PetscScalar     Fe[NODES_PER_EL*U_DOF];
    PetscScalar     el_coords[NODES_PER_EL*NSD];
    PetscScalar     ***ff;
    Vec             local_F;
    PetscInt        n,M,N,P;


    DMDAGetInfo(da,0,&M,&N,&P,0,0,0,0,0,0,0,0,0);
    /* setup for coords */
    DMGetCoordinateDM(da,&cda);
    DMGetCoordinatesLocal(da,&coords);
    DMDAVecGetArray(cda,coords,&_coords);

    /* get acces to the vector */
    DMGetLocalVector(da,&local_F);
    VecZeroEntries(local_F);
    DMDAVecGetArray(da,local_F,&ff);
    DMDAGetElementsCorners(da,&sex,&sey,&sez);
    DMDAGetElementsSizes(da,&mx,&my,&mz);

    for (ek = sez; ek < sez+mz; ek++) {
        for (ej = sey; ej < sey + my; ej++) {
            for (ei = sex; ei < sex + mx; ei++) {
                /* get coords for the element */
                GetElementCoords3D(_coords, ei, ej, ek, el_coords);

                /* initialise element stiffness matrix */
                PetscMemzero(Fe, sizeof(Fe));

                /* form element stiffness matrix */
                FormRHSQ13D(Fe,el_coords);

                /* insert element matrix into global matrix */
                DMDAGetElementEqnums3D_up(u_eqn,ei,ej,ek);

                for (n=0; n<NODES_PER_EL; n++) {
                    if ((u_eqn[n].i == 0) || (u_eqn[n].i == M-1)) Fe[n] = 0.0;

                    if ((u_eqn[n].j == 0) || (u_eqn[n].j == N-1)) Fe[n] = 0.0;

                    if ((u_eqn[n].k == 0) || (u_eqn[n].k == P-1)) Fe[n] = 0.0;
                }

                DMDASetValuesLocalStencil3D_ADD_VALUES(ff,u_eqn,Fe);
            }
        }
    }

    DMDAVecRestoreArray(da,local_F,&ff);
    DMLocalToGlobalBegin(da,local_F,ADD_VALUES,F);
    DMLocalToGlobalEnd(da,local_F,ADD_VALUES,F);

    DMDAVecRestoreArray(cda,coords,&_coords);

    return 0;
}

int main(int argc,char **args)
{
    KSP             ksp;
    DM              da;
    Mat             A;
    Vec             F,x;
    PetscInt        nel=3;
    PetscErrorCode  ierr;


    PetscInitialize(&argc,&args,(char*)nullptr,help);;
    PetscOptionsGetInt(nullptr,nullptr,"-nel",&nel,nullptr);
    DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,
                    nel+1,nel+1,nel+1,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,nullptr,nullptr,nullptr,&da);
    DMSetMatType(da,MATAIJ);
    DMSetFromOptions(da);
    DMSetUp(da);
    DMDASetUniformCoordinates(da,0,1,0,1,0,1);
    DMDASetFieldName(da,0,"Pressure");

    /* Generate a matrix with the correct non-zero pattern of type AIJ. This will work in parallel and serial */
    DMCreateMatrix(da,&A);
    DMCreateGlobalVector(da,&x);
    DMCreateGlobalVector(da,&F);

    /* assemble A11 */
    MatZeroEntries(A);
    VecZeroEntries(F);

    AssembleA_Poisson(A,da);
    MatViewFromOptions(A,nullptr,"-amat_view");

    AssembleF_Poisson(F,da);

    /* SOLVE */
    KSPCreate(PETSC_COMM_WORLD,&ksp);
    KSPSetOperators(ksp,A,A);
    KSPSetFromOptions(ksp);

    KSPSolve(ksp,F,x);
//    VecView(x,PETSC_VIEWER_STDOUT_WORLD);

    /* verify */
    DM  da_analytic;
    Vec x_analytic;

    DMDACreateManufacturedSolution(nel,nel,nel,&da_analytic,&x_analytic);

    DMDAIntegrateErrors3D(da_analytic,x,x_analytic);

    DMDestroy(&da_analytic);
    VecDestroy(&x_analytic);

    KSPDestroy(&ksp);
    VecDestroy(&x);
    VecDestroy(&F);
    MatDestroy(&A);

    DMDestroy(&da);

    PetscFinalize();
    return 0;

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

