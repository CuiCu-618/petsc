/*
    Concepts: KSP^solving a system of linear equations
    Concepts: KSP^Elastic, 3d
    Processors: n
 */

// FEM 3D elastic
// without SGFEM

static char help[] = "  Solves the elasticity equations in 3d on the unit domain using SGFEM. \n\
Options: \n"
"\
        -mx : number of elements in x-direction \n\
        -my : number of elements in y-direction \n\
        -mz : number of elements in z-direction \n\
Parameters: \n\
        -iso_E  : Youngs modulus \n\
        -iso_nu : Poisson ratio \n\n";

/* CU CUI */

#include <iostream>
#include <petscksp.h>
#include <petscdmda.h>
#include <petscdm.h>

static PetscErrorCode DMDABCApplyCompression(DM,Mat,Vec);

/* FEM basic info */
#define NSD            3 /* number of spatial dimensions */
#define NODES_PER_EL   8 /* nodes per element */
#define U_DOFS         3 /* dofs per node */
#define GAUSS_POINTS   8

/* SGFEM basic info */
#define Q       1 /* degree of Herminte PU */
#define L1      4 /* number of Heaviside enrichment */
#define L2      4 /* number of singular enrichment */
#define type    1 /* type=1: SGFEM; type=0: GFEM */

/* cell based evaluation */
typedef struct {
    PetscScalar E,nu,fx,fy,fz;
    PetscScalar gx,gy,gz;       /* derivative */
} Coefficients;

/* Gauss point based evaluation*/
typedef struct {
    PetscScalar gp_coords[NSD*GAUSS_POINTS];
    PetscScalar E[GAUSS_POINTS];
    PetscScalar nu[GAUSS_POINTS];
    PetscScalar fx[GAUSS_POINTS];
    PetscScalar fy[GAUSS_POINTS];
    PetscScalar fz[GAUSS_POINTS];
    PetscScalar gx[GAUSS_POINTS];
    PetscScalar gy[GAUSS_POINTS];
    PetscScalar gz[GAUSS_POINTS];
} GaussPointCoefficients;

typedef struct {
    PetscScalar ux_dof;
    PetscScalar uy_dof;
    PetscScalar uz_dof;
} ElasticityDOF;

/*

 D = E(1-nu)/((1+nu)(1-2nu)) * [    1     nu/(1-nu) nu/(1-nu)       0                   0                 0        ]
                               [nu/(1-nu)     1     nu/(1-nu)       0                   0                 0        ]
                               [nu/(1-nu)     1     nu/(1-nu)       1                   0                 0        ]
                               [    0         0         0     (1-2*nu)/(2-2*nu)         0                 0        ]
                               [    0         0         0           0           (1-2*nu)/(2-2*nu)         0        ]
                               [    0         0         0           0                   0         (1-2*nu)/(2-2*nu)]

 B = [ d_dx   0     0   ]
     [  0    d_dy   0   ]
     [  0     0    d_dz ]
     [ d_dy  d_dx   0   ]
     [  0    d_dz  d_dy ]
     [ d_dz   0    d_dx ]

 */


/* FEM routines */
/*
 Element: Local basis function ordering
 1-----2
 |     |
 |     |
 0-----3

   6-----7
  /|    /|
 1-----2 |
 | 5---|-8
 |/    |/
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

/* derivative on the reference cell */
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

/* derivative on the real cell */
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
 i,j,k are the element indices
 The unknown is a vector quantity.
 The s[].c is used to indicate the degree of freedom.
 */
static PetscErrorCode DMDAGetElementEqnums3D_u(MatStencil s_u[],PetscInt i,PetscInt j,PetscInt k)
{
    PetscInt n;

    PetscFunctionBeginUser;
    /* u */
    n = 0;
    /* node 0 */
    s_u[n].i = i; s_u[n].j = j; s_u[n].k = k; s_u[n].c = 0; n++; /* Ux0 */
    s_u[n].i = i; s_u[n].j = j; s_u[n].k = k; s_u[n].c = 1; n++; /* Uy0 */
    s_u[n].i = i; s_u[n].j = j; s_u[n].k = k; s_u[n].c = 2; n++; /* Uz0 */

    s_u[n].i = i; s_u[n].j = j+1; s_u[n].k = k; s_u[n].c = 0; n++;
    s_u[n].i = i; s_u[n].j = j+1; s_u[n].k = k; s_u[n].c = 1; n++;
    s_u[n].i = i; s_u[n].j = j+1; s_u[n].k = k; s_u[n].c = 2; n++;

    s_u[n].i = i+1; s_u[n].j = j+1; s_u[n].k = k; s_u[n].c = 0; n++;
    s_u[n].i = i+1; s_u[n].j = j+1; s_u[n].k = k; s_u[n].c = 1; n++;
    s_u[n].i = i+1; s_u[n].j = j+1; s_u[n].k = k; s_u[n].c = 2; n++;

    s_u[n].i = i+1; s_u[n].j = j; s_u[n].k = k; s_u[n].c = 0; n++;
    s_u[n].i = i+1; s_u[n].j = j; s_u[n].k = k; s_u[n].c = 1; n++;
    s_u[n].i = i+1; s_u[n].j = j; s_u[n].k = k; s_u[n].c = 2; n++;

    /* */
    s_u[n].i = i; s_u[n].j = j; s_u[n].k = k+1; s_u[n].c = 0; n++; /* Ux4 */
    s_u[n].i = i; s_u[n].j = j; s_u[n].k = k+1; s_u[n].c = 1; n++; /* Uy4 */
    s_u[n].i = i; s_u[n].j = j; s_u[n].k = k+1; s_u[n].c = 2; n++; /* Uz4 */

    s_u[n].i = i; s_u[n].j = j+1; s_u[n].k = k+1; s_u[n].c = 0; n++;
    s_u[n].i = i; s_u[n].j = j+1; s_u[n].k = k+1; s_u[n].c = 1; n++;
    s_u[n].i = i; s_u[n].j = j+1; s_u[n].k = k+1; s_u[n].c = 2; n++;

    s_u[n].i = i+1; s_u[n].j = j+1; s_u[n].k = k+1; s_u[n].c = 0; n++;
    s_u[n].i = i+1; s_u[n].j = j+1; s_u[n].k = k+1; s_u[n].c = 1; n++;
    s_u[n].i = i+1; s_u[n].j = j+1; s_u[n].k = k+1; s_u[n].c = 2; n++;

    s_u[n].i = i+1; s_u[n].j = j; s_u[n].k = k+1; s_u[n].c = 0; n++;
    s_u[n].i = i+1; s_u[n].j = j; s_u[n].k = k+1; s_u[n].c = 1; n++;
    s_u[n].i = i+1; s_u[n].j = j; s_u[n].k = k+1; s_u[n].c = 2; n++;

    PetscFunctionReturn(0);
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

/* get the nodal fields for u on one cell for error analysis
 * ElasticityDOF ***field stores the value at node (i,j,k)
 * ElasticityDOF nodal_fields[] converts ***field into a vector
 */
static PetscErrorCode ElasticDAGetNodalFields3D(ElasticityDOF ***field,PetscInt i,PetscInt j,PetscInt k,ElasticityDOF nodal_fields[])
{
    PetscFunctionBeginUser;
    /* get the nodal fields for u */
    nodal_fields[0].ux_dof = field[k][j][i].ux_dof;
    nodal_fields[0].uy_dof = field[k][j][i].uy_dof;
    nodal_fields[0].uz_dof = field[k][j][i].uz_dof;

    nodal_fields[1].ux_dof = field[k][j+1][i].ux_dof;
    nodal_fields[1].uy_dof = field[k][j+1][i].uy_dof;
    nodal_fields[1].uz_dof = field[k][j+1][i].uz_dof;

    nodal_fields[2].ux_dof = field[k][j+1][i+1].ux_dof;
    nodal_fields[2].uy_dof = field[k][j+1][i+1].uy_dof;
    nodal_fields[2].uz_dof = field[k][j+1][i+1].uz_dof;

    nodal_fields[3].ux_dof = field[k][j][i+1].ux_dof;
    nodal_fields[3].uy_dof = field[k][j][i+1].uy_dof;
    nodal_fields[3].uz_dof = field[k][j][i+1].uz_dof;

    nodal_fields[4].ux_dof = field[k+1][j][i].ux_dof;
    nodal_fields[4].uy_dof = field[k+1][j][i].uy_dof;
    nodal_fields[4].uz_dof = field[k+1][j][i].uz_dof;

    nodal_fields[5].ux_dof = field[k+1][j+1][i].ux_dof;
    nodal_fields[5].uy_dof = field[k+1][j+1][i].uy_dof;
    nodal_fields[5].uz_dof = field[k+1][j+1][i].uz_dof;

    nodal_fields[6].ux_dof = field[k+1][j+1][i+1].ux_dof;
    nodal_fields[6].uy_dof = field[k+1][j+1][i+1].uy_dof;
    nodal_fields[6].uz_dof = field[k+1][j+1][i+1].uz_dof;

    nodal_fields[7].ux_dof = field[k+1][j][i+1].ux_dof;
    nodal_fields[7].uy_dof = field[k+1][j][i+1].uy_dof;
    nodal_fields[7].uz_dof = field[k+1][j][i+1].uz_dof;

    PetscFunctionReturn(0);
}

/* add local Fe to global F
 * II,J,K global position
 */
static PetscErrorCode DMDASetValuesLocalStencil3D_ADD_VALUES(ElasticityDOF ***fields_F,MatStencil u_eqn[],PetscScalar Fe_u[])
{
    PetscInt n,II,J,K;

    PetscFunctionBeginUser;
    for (n = 0; n<NODES_PER_EL; n++) {
        II = u_eqn[NSD*n].i;
        J = u_eqn[NSD*n].j;
        K = u_eqn[NSD*n].k;

        fields_F[K][J][II].ux_dof = fields_F[K][J][II].ux_dof+Fe_u[NSD*n];

        II = u_eqn[NSD*n+1].i;
        J = u_eqn[NSD*n+1].j;
        K = u_eqn[NSD*n+1].k;

        fields_F[K][J][II].uy_dof = fields_F[K][J][II].uy_dof+Fe_u[NSD*n+1];

        II = u_eqn[NSD*n+2].i;
        J = u_eqn[NSD*n+2].j;
        K = u_eqn[NSD*n+2].k;
        fields_F[K][J][II].uz_dof = fields_F[K][J][II].uz_dof+Fe_u[NSD*n+2];

    }
    PetscFunctionReturn(0);
}

static void FormStressOperatorQ13D(PetscScalar Ke[],PetscScalar coords[],PetscScalar E[],PetscScalar nu[])
{
    PetscInt    ngp;
    PetscScalar gp_xi[GAUSS_POINTS][NSD];
    PetscScalar gp_weight[GAUSS_POINTS];
    PetscInt    p,i,j,k,l;
    PetscScalar GNi_p[NSD][NODES_PER_EL],GNx_p[NSD][NODES_PER_EL];
    PetscScalar J_p;
    PetscScalar B[6][U_DOFS*NODES_PER_EL];
    PetscScalar prop_E,prop_nu,factor,constit_D[6][6];
    const PetscInt nvdof = U_DOFS*NODES_PER_EL;

    /* define quadrature rule */
    ConstructGaussQuadrature3D(&ngp,gp_xi,gp_weight);

    /* evaluate integral */
    for (p = 0; p < ngp; p++) {
        ShapeFunctionQ13D_Evaluate_dxi(gp_xi[p],GNi_p);
        ShapeFunctionQ13D_Evaluate_dx(GNi_p,GNx_p,coords,&J_p);

        for (i = 0; i < NODES_PER_EL; i++) {
            PetscScalar d_dx_i = GNx_p[0][i];
            PetscScalar d_dy_i = GNx_p[1][i];
            PetscScalar d_dz_i = GNx_p[2][i];

            B[0][3*i] = d_dx_i; B[0][3*i+1] = 0.0;     B[0][3*i+2] = 0.0;
            B[1][3*i] = 0.0;    B[1][3*i+1] = d_dy_i;  B[1][3*i+2] = 0.0;
            B[2][3*i] = 0.0;    B[2][3*i+1] = 0.0;     B[2][3*i+2] = d_dz_i;

            B[3][3*i] = d_dy_i; B[3][3*i+1] = d_dx_i;  B[3][3*i+2] = 0.0;   /* e_xy */
            B[4][3*i] = 0.0;    B[4][3*i+1] = d_dz_i;  B[4][3*i+2] = d_dy_i; /* e_yz */
            B[5][3*i] = d_dz_i; B[5][3*i+1] = 0.0;     B[5][3*i+2] = d_dx_i; /* e_zx */
        }

        /* form D for the quadrature point */
        prop_E          = E[p];
        prop_nu         = nu[p];
        factor          = prop_E / ((1.0+prop_nu)*(1.0-2.0*prop_nu));

        PetscMemzero(constit_D, sizeof(constit_D));
        constit_D[0][0] = 1.0; constit_D[0][1] = prop_nu/(1-prop_nu); constit_D[0][2] = prop_nu/(1-prop_nu);
        constit_D[1][0] = prop_nu/(1-prop_nu); constit_D[1][1] = 1.0; constit_D[1][2] = prop_nu/(1-prop_nu);
        constit_D[2][0] = prop_nu/(1-prop_nu); constit_D[2][1] = prop_nu/(1-prop_nu); constit_D[2][2] = 1.0;
        constit_D[3][3] = 0.5*(1-2*prop_nu)/(1-prop_nu);
        constit_D[4][4] = 0.5*(1-2*prop_nu)/(1-prop_nu);
        constit_D[5][5] = 0.5*(1-2*prop_nu)/(1-prop_nu);
        for (i = 0; i < 6; i++) {
            for (j = 0; j < 6; j++) {
                constit_D[i][j] = factor * constit_D[i][j] * gp_weight[p] * J_p;
            }
        }

        /* form Bt tildeD B */
        /*
         Ke_ij = Bt_ik . D_kl . B_lj
         = B_ki . D_kl . B_lj
         */
        for (i = 0; i < nvdof; i++) {
            for (j = 0; j < nvdof; j++) {
                for (k = 0; k < 6; k++) {
                    for (l = 0; l < 6; l++) {
                        Ke[nvdof*i+j] = Ke[nvdof*i+j] + B[k][i] * constit_D[k][l] * B[l][j];
                    }
                }
            }
        }

    } /* end quadrature */
}

static void FormMomentumRhsQ13D(PetscScalar Fe[],PetscScalar coords[],PetscScalar fx[],PetscScalar fy[],PetscScalar fz[])
{
    PetscInt    ngp;
    PetscScalar gp_xi[GAUSS_POINTS][NSD];
    PetscScalar gp_weight[GAUSS_POINTS];
    PetscInt    p,i;
    PetscScalar Ni_p[NODES_PER_EL];
    PetscScalar GNi_p[NSD][NODES_PER_EL],GNx_p[NSD][NODES_PER_EL];
    PetscScalar J_p,fac;

    /* define quadrature rule */
    ConstructGaussQuadrature3D(&ngp,gp_xi,gp_weight);

    /* evaluate integral */
    for (p = 0; p < ngp; p++) {
        ShapeFunctionQ13D_Evaluate(gp_xi[p],Ni_p);
        ShapeFunctionQ13D_Evaluate_dxi(gp_xi[p],GNi_p);
        ShapeFunctionQ13D_Evaluate_dx(GNi_p,GNx_p,coords,&J_p);
        fac = gp_weight[p]*J_p;

        for (i = 0; i < NODES_PER_EL; i++) {
            Fe[NSD*i]   += fac*Ni_p[i]*fx[p];
            Fe[NSD*i+1] += fac*Ni_p[i]*fy[p];
            Fe[NSD*i+2] += fac*Ni_p[i]*fz[p];
        }
    }
}

static PetscErrorCode AssembleA_Elasticity(Mat A,DM elas_da,DM properties_da,Vec properties)
{
    DM                     cda;
    Vec                    coords;
    DMDACoor3d             ***_coords;
    MatStencil             u_eqn[NODES_PER_EL*U_DOFS]; /* 3 degrees of freedom */
    PetscInt               sex,sey,sez,mx,my,mz;
    PetscInt               ei,ej,ek;
    PetscScalar            Ae[NODES_PER_EL*U_DOFS*NODES_PER_EL*U_DOFS];
    PetscScalar            el_coords[NODES_PER_EL*NSD];
    Vec                    local_properties;
    GaussPointCoefficients ***props;
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

    ierr = DMDAGetElementsCorners(elas_da,&sex,&sey,&sez);CHKERRQ(ierr);
    ierr = DMDAGetElementsSizes(elas_da,&mx,&my,&mz);CHKERRQ(ierr);
    for (ek = sez; ek < sez+mz; ek++) {
        for (ej = sey; ej < sey + my; ej++) {
            for (ei = sex; ei < sex + mx; ei++) {
                /* get coords for the element */
                GetElementCoords3D(_coords, ei, ej, ek, el_coords);

                /* get coefficients for the element */
                prop_E = props[ek][ej][ei].E;
                prop_nu = props[ek][ej][ei].nu;

                /* initialise element stiffness matrix */
                ierr = PetscMemzero(Ae, sizeof(Ae));CHKERRQ(ierr);

                /* form element stiffness matrix */
                FormStressOperatorQ13D(Ae, el_coords, prop_E, prop_nu);

                /* insert element matrix into global matrix */
                ierr = DMDAGetElementEqnums3D_u(u_eqn, ei, ej, ek);CHKERRQ(ierr);
                ierr = MatSetValuesStencil(A, NODES_PER_EL * U_DOFS, u_eqn, NODES_PER_EL * U_DOFS, u_eqn, Ae,
                                           ADD_VALUES);CHKERRQ(ierr);
            }
        }
    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    ierr = DMDAVecRestoreArray(cda,coords,&_coords);CHKERRQ(ierr);

    ierr = DMDAVecRestoreArray(properties_da,local_properties,&props);CHKERRQ(ierr);
    ierr = VecDestroy(&local_properties);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

static PetscErrorCode AssembleF_Elasticity(Vec F,DM elas_da,DM properties_da,Vec properties)
{
    DM                     cda;
    Vec                    coords;
    DMDACoor3d             ***_coords;
    MatStencil             u_eqn[NODES_PER_EL*U_DOFS]; /* 3 degrees of freedom */
    PetscInt               sex,sey,sez,mx,my,mz;
    PetscInt               ei,ej,ek;
    PetscScalar            Fe[NODES_PER_EL*U_DOFS];
    PetscScalar            el_coords[NODES_PER_EL*NSD];
    Vec                    local_properties;
    GaussPointCoefficients ***props;
    PetscScalar            *prop_fx,*prop_fy,*prop_fz;
    Vec                    local_F;
    ElasticityDOF          ***ff;
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

    ierr = DMDAGetElementsCorners(elas_da,&sex,&sey,&sez);CHKERRQ(ierr);
    ierr = DMDAGetElementsSizes(elas_da,&mx,&my,&mz);CHKERRQ(ierr);
    for (ek = sez; ek < sez+mz; ek++) {
        for (ej = sey; ej < sey + my; ej++) {
            for (ei = sex; ei < sex + mx; ei++) {
                /* get coords for the element */
                GetElementCoords3D(_coords, ei, ej, ek, el_coords);

                /* get coefficients for the element */
                prop_fx = props[ek][ej][ei].fx;
                prop_fy = props[ek][ej][ei].fy;
                prop_fz = props[ek][ej][ei].fz;

                /* initialise element stiffness matrix */
                ierr = PetscMemzero(Fe, sizeof(Fe));CHKERRQ(ierr);

                /* form element stiffness matrix */
                FormMomentumRhsQ13D(Fe, el_coords, prop_fx, prop_fy, prop_fz);

                /* insert element matrix into global matrix */
                ierr = DMDAGetElementEqnums3D_u(u_eqn, ei, ej, ek);CHKERRQ(ierr);

                ierr = DMDASetValuesLocalStencil3D_ADD_VALUES(ff, u_eqn, Fe);CHKERRQ(ierr);
            }
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

static PetscErrorCode DMDAGetElementOwnershipRanges3d(DM da,PetscInt **_lx,PetscInt **_ly,PetscInt **_lz)
{
    PetscErrorCode ierr;
    PetscMPIInt    rank;
    PetscInt       proc_I,proc_J,proc_K;
    PetscInt       cpu_x,cpu_y,cpu_z;
    PetscInt       local_mx,local_my,local_mz;
    Vec            vlx,vly,vlz;
    PetscInt       *LX,*LY,*LZ,i;
    PetscScalar    *_a;
    Vec            V_SEQ;
    VecScatter     ctx;

    PetscFunctionBeginUser;
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

    DMDAGetInfo(da,0,0,0,0,&cpu_x,&cpu_y,&cpu_z,0,0,0,0,0,0);

    proc_K = rank/(cpu_x*cpu_y);
    proc_J = (rank-cpu_x*cpu_y*proc_K)/cpu_x;
    proc_I = rank-cpu_x*cpu_y*proc_K-cpu_x*proc_J;

    ierr = PetscMalloc1(cpu_x,&LX);CHKERRQ(ierr);
    ierr = PetscMalloc1(cpu_y,&LY);CHKERRQ(ierr);
    ierr = PetscMalloc1(cpu_x,&LZ);CHKERRQ(ierr);

    ierr = DMDAGetElementsSizes(da,&local_mx,&local_my,&local_mz);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_WORLD,&vlx);CHKERRQ(ierr);
    ierr = VecSetSizes(vlx,PETSC_DECIDE,cpu_x);CHKERRQ(ierr);
    ierr = VecSetFromOptions(vlx);CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD,&vly);CHKERRQ(ierr);
    ierr = VecSetSizes(vly,PETSC_DECIDE,cpu_y);CHKERRQ(ierr);
    ierr = VecSetFromOptions(vly);CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD,&vlz);CHKERRQ(ierr);
    ierr = VecSetSizes(vlz,PETSC_DECIDE,cpu_z);CHKERRQ(ierr);
    ierr = VecSetFromOptions(vlz);CHKERRQ(ierr);

    ierr = VecSetValue(vlx,proc_I,(PetscScalar)(local_mx+1.0e-9),INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(vly,proc_J,(PetscScalar)(local_my+1.0e-9),INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(vlz,proc_K,(PetscScalar)(local_mz+1.0e-9),INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(vlx);VecAssemblyEnd(vlx);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(vly);VecAssemblyEnd(vly);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(vlz);VecAssemblyEnd(vlz);CHKERRQ(ierr);

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

    ierr = VecScatterCreateToAll(vlz,&ctx,&V_SEQ);CHKERRQ(ierr);
    ierr = VecScatterBegin(ctx,vlz,V_SEQ,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ctx,vlz,V_SEQ,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray(V_SEQ,&_a);CHKERRQ(ierr);
    for (i = 0; i < cpu_z; i++) LZ[i] = (PetscInt)PetscRealPart(_a[i]);
    ierr = VecRestoreArray(V_SEQ,&_a);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);
    ierr = VecDestroy(&V_SEQ);CHKERRQ(ierr);

    *_lx = LX;
    *_ly = LY;
    *_lz = LZ;

    ierr = VecDestroy(&vlx);CHKERRQ(ierr);
    ierr = VecDestroy(&vly);CHKERRQ(ierr);
    ierr = VecDestroy(&vlz);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

static PetscErrorCode solve_elasticity_3d(PetscInt mx, PetscInt my, PetscInt mz)
{
    DM                     elas_da,da_prop;
    PetscInt               u_dof,dof,stencil_width;
    Mat                    A;
    PetscInt               mxl,myl,mzl;
    DM                     prop_cda,vel_cda;
    Vec                    prop_coords,vel_coords;
    PetscInt               si,sj,sk,nx,ny,nz,i,j,k,p;
    Vec                    f,X;
    PetscInt               prop_dof,prop_stencil_width;
    Vec                    properties,l_properties;
    MatNullSpace           matnull;
    PetscReal              dx,dy,dz;
    PetscInt               M,N,P;
    DMDACoor3d             ***_prop_coords,***_vel_coords;
    GaussPointCoefficients ***element_props;
    KSP                    ksp_E;
    PetscInt               coefficient_structure = 0;
    PetscInt               cpu_x,cpu_y,cpu_z,*lx = nullptr,*ly = nullptr,*lz = nullptr;
    PetscBool              use_gp_coords = PETSC_FALSE;
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
    u_dof         = U_DOFS; /* Ux, Uy , Uz */
    dof           = u_dof;
    stencil_width = 1;
    ierr          = DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                                 DMDA_STENCIL_BOX, mx+1, my+1, mz+1, PETSC_DECIDE,PETSC_DECIDE, PETSC_DECIDE,
                                 dof, stencil_width, nullptr, nullptr, nullptr, &elas_da);CHKERRQ(ierr);

    ierr = DMSetMatType(elas_da,MATAIJ);CHKERRQ(ierr);
    ierr = DMSetFromOptions(elas_da);CHKERRQ(ierr);
    ierr = DMSetUp(elas_da);CHKERRQ(ierr);

    ierr = DMDASetFieldName(elas_da,0,"Ux");CHKERRQ(ierr);
    ierr = DMDASetFieldName(elas_da,1,"Uy");CHKERRQ(ierr);
    ierr = DMDASetFieldName(elas_da,2,"Uz");CHKERRQ(ierr);

    /* unit box [0,1] x [0,1] x [0,1] */
    ierr = DMDASetUniformCoordinates(elas_da,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);

    /* Generate element properties, we will assume all material properties are constant over the element */
    /* local number of elements */
    ierr = DMDAGetElementsSizes(elas_da,&mxl,&myl,&mzl);CHKERRQ(ierr);

    /* !!! IN PARALLEL WE MUST MAKE SURE THE TWO DMDA's ALIGN !!! */
    ierr = DMDAGetInfo(elas_da,0,0,0,0,&cpu_x,&cpu_y,&cpu_z,0,0,0,0,0,0);CHKERRQ(ierr);
    ierr = DMDAGetElementOwnershipRanges3d(elas_da,&lx,&ly,&lz);CHKERRQ(ierr);

    prop_dof           = (PetscInt)(sizeof(GaussPointCoefficients)/sizeof(PetscScalar)); /* gauss point setup */
    prop_stencil_width = 0;
    ierr               = DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                                      DMDA_STENCIL_BOX,mx,my,mz,cpu_x,cpu_y,cpu_z,prop_dof,prop_stencil_width,
                                      lx,ly,lz,&da_prop);CHKERRQ(ierr);
    ierr = DMSetFromOptions(da_prop);CHKERRQ(ierr);
    ierr = DMSetUp(da_prop);CHKERRQ(ierr);

    ierr = PetscFree(lx);CHKERRQ(ierr);
    ierr = PetscFree(ly);CHKERRQ(ierr);

    /* define centroid positions */
    ierr = DMDAGetInfo(da_prop,0,&M,&N,&P,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
    dx   = 1.0/((PetscReal)(M));
    dy   = 1.0/((PetscReal)(N));
    dz   = 1.0/((PetscReal)(P));

    ierr = DMDASetUniformCoordinates(da_prop,0.0+0.5*dx,1.0-0.5*dx,0.0+0.5*dy,1.0-0.5*dy,0.0+0.5*dz,1.0-0.5*dz);CHKERRQ(ierr);

    ierr = DMCreateGlobalVector(da_prop,&properties);CHKERRQ(ierr);
    ierr = DMCreateLocalVector(da_prop,&l_properties);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da_prop,l_properties,&element_props);CHKERRQ(ierr);

    ierr = DMGetCoordinateDM(da_prop,&prop_cda);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(da_prop,&prop_coords);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(prop_cda,prop_coords,&_prop_coords);CHKERRQ(ierr);

    ierr = DMDAGetGhostCorners(prop_cda,&si,&sj,&sk,&nx,&ny,&nz);CHKERRQ(ierr);

    ierr = DMGetCoordinateDM(elas_da,&vel_cda);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(elas_da,&vel_coords);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(vel_cda,vel_coords,&_vel_coords);CHKERRQ(ierr);

    /* interpolate the coordinates to quadrature points */
    for (k = sk; k < sk+nz; k++) {
        for (j = sj; j < sj+ny; j++) {
            for (i = si; i < si+nx; i++) {
                /* get coords for the element */
                PetscInt               ngp;
                PetscScalar            gp_xi[GAUSS_POINTS][NSD],gp_weight[GAUSS_POINTS];
                PetscScalar            el_coords[NSD*NODES_PER_EL];

                ierr = GetElementCoords3D(_vel_coords,i,j,k,el_coords);CHKERRQ(ierr);
                ConstructGaussQuadrature3D(&ngp,gp_xi,gp_weight);

                for (p = 0; p < GAUSS_POINTS; p++) {
                    PetscScalar xi_p[NSD],Ni_p[NODES_PER_EL];
                    PetscScalar gp_x,gp_y,gp_z;
                    PetscInt    n;

                    xi_p[0] = gp_xi[p][0];
                    xi_p[1] = gp_xi[p][1];
                    xi_p[2] = gp_xi[p][2];
                    ShapeFunctionQ13D_Evaluate(xi_p,Ni_p);

                    gp_x = gp_y = gp_z = 0.0;
                    for (n = 0; n < NODES_PER_EL; n++) {
                        gp_x = gp_x+Ni_p[n]*el_coords[NSD*n];
                        gp_y = gp_y+Ni_p[n]*el_coords[NSD*n+1];
                        gp_z = gp_z+Ni_p[n]*el_coords[NSD*n+2];
                    }
                    element_props[k][j][i].gp_coords[NSD*p]   = gp_x;
                    element_props[k][j][i].gp_coords[NSD*p+1] = gp_y;
                    element_props[k][j][i].gp_coords[NSD*p+2] = gp_z;
                }
            }
        }
    }
    for (k = sk; k < sk+nz; k++) {
        for (j = sj; j < sj + ny; j++) {
            for (i = si; i < si + nx; i++) {
                PetscScalar coord_x, coord_y, coord_z;

                PetscScalar opts_E, opts_nu;

                opts_E = 1.0;
                opts_nu = 0.33;
                ierr = PetscOptionsGetScalar(NULL, NULL, "-iso_E", &opts_E, &flg);CHKERRQ(ierr);
                ierr = PetscOptionsGetScalar(NULL, NULL, "-iso_nu", &opts_nu, &flg);CHKERRQ(ierr);

                for (p = 0; p < GAUSS_POINTS; p++) {
                    coord_x = element_props[k][j][i].gp_coords[NSD*p];
                    coord_y = element_props[k][j][i].gp_coords[NSD*p+1];
                    coord_z = element_props[k][j][i].gp_coords[NSD*p+2];

                    element_props[k][j][i].E[p] = opts_E;
                    element_props[k][j][i].nu[p] = opts_nu;

                    element_props[k][j][i].fx[p] = 0.0;
                    element_props[k][j][i].fy[p] = 0.0;
                    element_props[k][j][i].fz[p] = 0.0;

                    element_props[k][j][i].gx[p] = 0.0;
                    element_props[k][j][i].gy[p] = 0.0;
                    element_props[k][j][i].gz[p] = 0.0;
                }
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

    /* assemble All */
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
    PetscInt        mx,my,mz;

    ierr = PetscInitialize(&argc,&args,(char*)0,help); if (ierr) return ierr;
    mx   = my = mz = 5;
    ierr = PetscOptionsGetInt(NULL,NULL,"-mx",&mx,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-my",&mx,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-mz",&mx,NULL);CHKERRQ(ierr);

    ierr = solve_elasticity_3d(mx,my,mz);CHKERRQ(ierr);
    ierr = PetscFinalize();

    return ierr;
}


/* -------------------------- helpers for boundary conditions -------------------------------- */
static PetscErrorCode DMDABCApplyCompression(DM elas_da,Mat A,Vec f)
{
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    PetscFunctionReturn(0);
}