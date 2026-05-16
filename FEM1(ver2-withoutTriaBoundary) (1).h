/* Очищенный файл FEM1.h */

/*This is a template file for use with 1D finite elements.
  The portions of the code you need to fill in are marked with the comment "//EDIT".

  Do not change the name of any existing functions, but feel free
  to create additional functions, variables, and constants.
  It uses the deal.II FEM library.*/

//Include files
//Data structures and solvers
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
//Mesh related classes
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
//#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
//Finite element implementation classes
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
//Standard C++ libraries
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>

using namespace dealii;

template <int dim>
class FEM
{
 public:
  //Class functions
  FEM (unsigned int order,unsigned int problem); // Class constructor 
  ~FEM(); //Class destructor

  //Function to find the value of xi at the given node (using deal.II node numbering)
  double xi_at_node(unsigned int dealNode);

  //Define your 1D basis functions and derivatives
  double basis_function(unsigned int node, double xi);
  double basis_gradient(unsigned int node, double xi);

  //Solution steps
  void generate_mesh(unsigned int numberOfElements); // разделяет отрезок [0;L] на равные части.
  void define_boundary_conds(); // заполняет boundary_values словарь (либо правый конец будет свободен, либо нет).
  void setup_system();
  // распределеляет степени свободы dof_handler.distribute_dofs,
  // указывает координаты узлов nodeLocation,
  // вызывает функцию define_boundary_conds,
  // создаёт K матрицу, F и D векторы,
  // задание квадратурные точки (опорных узлов) + их весов.
  void assemble_system(); // собирает KD = F.
  void solve(); // решает KD = F.
  void output_results(); // создаёт solution.vtk, где указаны значения перемещений.

  //Function to calculate the l2 norm of the error in the finite element sol'n vs. the exact solution
  double l2norm_of_error(); // возвращает sqrt(инт.(u - uh)^2*dx)

  //Class objects
  Triangulation<dim>   triangulation; //mesh
  FESystem<dim>        fe;           //FE element
  DoFHandler<dim>      dof_handler;   //Connectivity matrices

  //Gaussian quadrature - These will be defined in setup_system()
  unsigned int          quadRule;    //quadrature rule, i.e. number of quadrature points
  std::vector<double>   quad_points; //vector of Gauss quadrature points
  std::vector<double>   quad_weight; //vector of the quadrature point weights
    
  //Data structures
  SparsityPattern       sparsity_pattern; //Sparse matrix pattern
  SparseMatrix<double>  K;               //Global stiffness (sparse) matrix
  Vector<double>        D, F;            //Global vectors - Solution vector (D) and Global force vector (F)
  std::vector<double>   nodeLocation;    //Vector of the x-coordinate of nodes by global dof number
  std::map<unsigned int,double> boundary_values; //Map of dirichlet boundary conditions
  double                basisFunctionOrder, prob, L, g1, g2;
  double E, A, f_bar, h_val;

  //solution name array
  std::vector<std::string> nodal_solution_names;
  std::vector<DataComponentInterpretation::DataComponentInterpretation> nodal_data_component_interpretation;
};

// Class constructor for a vector field
template <int dim>
FEM<dim>::FEM(unsigned int order,unsigned int problem)
:
fe (FE_Q<dim>(order), dim), 
  dof_handler (triangulation)
{
  basisFunctionOrder = order;
  prob = (problem == 1 || problem == 2) ? problem : 1;

  E = 1e11;      
  A = 1e-4;      
  f_bar = 1e11;  
  h_val = 1e6;   
  L = 0.1; 
  g1 = 0.0;
  g2 = 0.001;

  for (unsigned int i=0; i<dim; ++i){
    nodal_solution_names.push_back("u");
    nodal_data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
  }
}

//Class destructor
template <int dim>
FEM<dim>::~FEM(){
  dof_handler.clear();
}

//Find the value of xi at the given node (using deal.II node numbering)
template <int dim>
double FEM<dim>::xi_at_node(unsigned int dealNode){ // та расстановка квадратурных точек (опорных узлов).
  double xi;

  if(dealNode == 0){
    xi = -1.;
  }
  else if(dealNode == 1){
    xi = 1.;
  }
  else if(dealNode <= basisFunctionOrder){
    xi = -1. + 2.*(dealNode-1.)/basisFunctionOrder;
  }
  else{
    std::cout << "Error: you input node number "
              << dealNode << " but there are only " 
              << basisFunctionOrder + 1 << " nodes in an element.\n";
    exit(0);
  }

  return xi;
}

//Define basis functions
template <int dim>
double FEM<dim>::basis_function(unsigned int node, double xi){
  // Насколько сильно какой-либо узел влияет на точку.
  /*
  Откуда берётся решение? Скажем, у нас есть узлы (традиционно, обозн. левый увел - 0, правый - 1):
  
  узел_0      узел_2        узел_1
  | ----------- | ----------- |

  Поставили точку:

  узел_0  точка  узел_2        узел_1
  | ------- | ---- | ----------- |  

  Теперь, хотим выяснить, какой "вес" (какой "процент" вклада в точку определяется конкретным узлом) оказывает узел_2 на точку.
  Итак, чтобы вычислить чисто влияние узла_2, вклад остальных придётся занулить.
  Делается это так:
        Узел_0 должен равняться 0, значит, при x=0 должен возвращаться 0:
    (x-0)
        Узел_1 должен также равняться 0. При x=1 будет 0:
        (x-1)
    Теперь, если x=0.5, то мы попали ровно на узел_2, и должна вернуться 1 (100% значения точки задаётся знанием узла).
        (x-0)/0.5 * (x-1)/-0.5.
  */
        
  /*"basisFunctionOrder" defines the polynomial order of the basis function,
    "node" specifies which node the basis function corresponds to, 
    "xi" is the point (in the bi-unit domain) where the function is being evaluated.
    You need to calculate the value of the specified basis function and order at the given quadrature pt.*/

  double value = 1.; //Store the value of the basis function in this variable
  // значение веса в конкретной точке, полученное от интерполяции, которое мы хотим найти.

  /*You can use the function "xi_at_node" (defined above) to get the value of xi (in the bi-unit domain)
    at any node in the element - using deal.II's element node numbering pattern.*/

  //EDIT DONE

  // Для текущего узла получим хи (X переводим в Ksi).
  double xi_node = xi_at_node(node);
  
  for(unsigned int B = 0; B <= basisFunctionOrder; B++){
    if(B != node){
      value *= (xi - xi_at_node(B)) / (xi_node - xi_at_node(B));
    }
  }

  return value;
}

//Define basis function gradient
template <int dim>
double FEM<dim>::basis_gradient(unsigned int node, double xi){
  // Функция, которая возвращает тот самый вес узла в какой-либо точке.

  /*"basisFunctionOrder" defines the polynomial order of the basis function,
    "node" specifies which node the basis function corresponds to, 
    "xi" is the point (in the bi-unit domain) where the function is being evaluated.
    You need to calculate the value of the derivative of the specified basis function and order at the given quadrature pt.
    Note that this is the derivative with respect to xi (not x)*/

  double value = 0.; //Store the value of the gradient of the basis function in this variable

  /*You can use the function "xi_at_node" (defined above) to get the value of xi (in the bi-unit domain)
    at any node in the element - using deal.II's element node numbering pattern.*/

  if(basisFunctionOrder == 1){ // Линейный
    if(node == 0){
      value = -0.5;
    }
    else if(node == 1){
      value = 0.5;
    }
  }
  else if(basisFunctionOrder == 2){ // Квадратичный
    if(node == 0){
      value = xi - 0.5;
    }
    else if(node == 1){
      value = xi + 0.5;
    }
    else if(node == 2){
      value = -2.0 * xi;
    }
  }
  else if(basisFunctionOrder == 3){ // Кубический
    if(node == 0){
      value = -(27.0*xi*xi - 18.0*xi - 1.0) / 16.0;
    }
    else if(node == 1){
      value = (27.0*xi*xi + 18.0*xi - 1.0) / 16.0;
    }
    else if(node == 2){
      value = 9.0*(9.0*xi*xi - 2.0*xi - 3.0) / 16.0;
    }
    else if(node == 3){
      value = -9.0*(9.0*xi*xi + 2.0*xi - 3.0) / 16.0;
    }
  }

  return value;
}

//Define the problem domain and generate the mesh
template <int dim>
void FEM<dim>::generate_mesh(unsigned int numberOfElements){

  //Define the limits of your domain
  L = 0.1; //EDIT DONE // 0.1 m
  double x_min = 0.;
  double x_max = L;

  Point<dim,double> min(x_min),
    max(x_max);
  std::vector<unsigned int> meshDimensions (dim,numberOfElements);
  GridGenerator::subdivided_hyper_rectangle (triangulation, meshDimensions, min, max);
}

//Specify the Dirichlet boundary conditions
template <int dim>
void FEM<dim>::define_boundary_conds(){
  const unsigned int totalNodes = dof_handler.n_dofs(); //Total number of nodes

  //Identify dirichlet boundary nodes and specify their values.
  //This function is called from within "setup_system"

  /*The vector "nodeLocation" gives the x-coordinate in the real domain of each node,
    organized by the global node number.*/

  /*This loops through all nodes in the system and checks to see if they are
    at one of the boundaries. If at a Dirichlet boundary, it stores the node number
    and the applied displacement value in the std::map "boundary_values". Deal.II
    will use this information later to apply Dirichlet boundary conditions.
    Neumann boundary conditions are applied when constructing Flocal in "assembly"*/
  for(unsigned int globalNode=0; globalNode<totalNodes; globalNode++){
    if(nodeLocation[globalNode] == 0){
      boundary_values[globalNode] = g1;
    }
    if(nodeLocation[globalNode] == L){
      if(prob == 1){
        boundary_values[globalNode] = g2;
      }
    }
  }
                        
}

//Setup data structures (sparse matrix, vectors)
template <int dim>
void FEM<dim>::setup_system(){

  //Define constants for problem (Dirichlet boundary values)

  // EDIT DONE

  //Let deal.II organize degrees of freedom
  dof_handler.distribute_dofs (fe);

  //Enter the global node x-coordinates into the vector "nodeLocation"
  MappingQ1<dim,dim> mapping;
  std::vector< Point<dim,double> > dof_coords(dof_handler.n_dofs());
  nodeLocation.resize(dof_handler.n_dofs());
  DoFTools::map_dofs_to_support_points<dim,dim>(mapping,dof_handler,dof_coords);
  for(unsigned int i=0; i<dof_coords.size(); i++){
    nodeLocation[i] = dof_coords[i][0];
  }

  //Specify boundary condtions (call the function)
  define_boundary_conds();

  //Define the size of the global matrices and vectors
  sparsity_pattern.reinit (dof_handler.n_dofs(), dof_handler.n_dofs(),
                           dof_handler.max_couplings_between_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);
  sparsity_pattern.compress();
  K.reinit (sparsity_pattern);
  F.reinit (dof_handler.n_dofs());
  D.reinit (dof_handler.n_dofs());

  //Define quadrature rule
  /*A quad rule of 2 is included here as an example. You will need to decide
    what quadrature rule is needed for the given problems*/

  // EDIT DONE
        
  if(basisFunctionOrder == 1){
    quadRule = 1;
  }
  else if(basisFunctionOrder == 2){
    quadRule = 2;
  }
  else if(basisFunctionOrder == 3){
    quadRule = 3;
  }

  quad_points.resize(quadRule);
  quad_weight.resize(quadRule);

  if(quadRule == 1){
    quad_points[0] = 0.0;
    quad_weight[0] = 2.0;
  }
  else if(quadRule == 2){
    quad_points[0] = -0.577350269189626;
    quad_points[1] =  0.577350269189626;
    quad_weight[0] = 1.0;
    quad_weight[1] = 1.0;
  }
  else if(quadRule == 3){
    quad_points[0] = -0.774596669241483;
    quad_points[1] =  0.0;
    quad_points[2] =  0.774596669241483;
    quad_weight[0] = 0.555555555555556;
    quad_weight[1] = 0.888888888888889;
    quad_weight[2] = 0.555555555555556;
  }

  std::cout<<"     Quad Order:      "<<quadRule<<std::endl;
  //Just some notes...
  std::cout << "   Number of active elems:       " << triangulation.n_active_cells() << std::endl;
  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;   
}

//Form elmental vectors and matrices and assemble to the global vector (F) and matrix (K)
template <int dim>
void FEM<dim>::assemble_system(){

  K=0; F=0;

  const unsigned int dofs_per_elem = fe.dofs_per_cell;
  FullMatrix<double> Klocal(dofs_per_elem, dofs_per_elem);
  Vector<double> Flocal(dofs_per_elem);
  std::vector<unsigned int> local_dof_indices(dofs_per_elem);
  double h_e, x, f;

  //loop over elements  
  typename DoFHandler<dim>::active_cell_iterator elem = dof_handler.begin_active(), 
    endc = dof_handler.end();
  for (;elem!=endc; ++elem){

    /*Retrieve the effective "connectivity matrix" for this element
      "local_dof_indices" relates local dofs to global dofs,
      i.e. local_dof_indices[i] gives the global dof number for local dof i.*/
    elem->get_dof_indices (local_dof_indices);

    /*We find the element length by subtracting the x-coordinates of the two end nodes
      of the element. Remember that the vector "nodeLocation" holds the x-coordinates, indexed
      by the global node number. "local_dof_indices" gives us the global node number indexed by
      the element node number.*/
    h_e = nodeLocation[local_dof_indices[1]] - nodeLocation[local_dof_indices[0]];

    //Loop over local DOFs and quadrature points to populate Flocal and Klocal.
    //EDIT: Использованы переменные класса (A, f_bar), чтобы изменения в main.cc учитывались
    Flocal = 0.;
    for(unsigned int A_idx=0; A_idx<dofs_per_elem; A_idx++){
      for(unsigned int q=0; q<quadRule; q++){
        x = 0;
        //Interpolate the x-coordinates at the nodes to find the x-coordinate at the quad pt.
        for(unsigned int B=0; B<dofs_per_elem; B++){
          x += nodeLocation[local_dof_indices[B]]*basis_function(B,quad_points[q]);
        }
        f = f_bar * x;  // f = ¯f * x согласно условию задания
        Flocal(A_idx) += f * basis_function(A_idx, quad_points[q]) * quad_weight[q] * (h_e / 2.0) * A;
      }
    }
    
    //Add nonzero Neumann condition, if applicable
    //EDIT: Использованы переменные h_val и L из класса
    if(prob == 2){ 
      // Проверка на близость к правой границе L (используем допуск 1e-9 для точности)
      if(std::abs(nodeLocation[local_dof_indices[dofs_per_elem-1]] - L) < 1e-9){
        //EDIT - Modify Flocal to include the traction on the right boundary.
        Flocal(dofs_per_elem-1) += h_val * A;
      }
    }

    //Loop over local DOFs and quadrature points to populate Klocal
    Klocal = 0;
    //EDIT: Использованы переменные E и A из класса
    for(unsigned int A_idx=0; A_idx<dofs_per_elem; A_idx++){
      for(unsigned int B_idx=0; B_idx<dofs_per_elem; B_idx++){
        for(unsigned int q=0; q<quadRule; q++){
          //EDIT DONE - Define Klocal.
          double dphiA_dx = basis_gradient(A_idx, quad_points[q]) * (2.0 / h_e);
          double dphiB_dx = basis_gradient(B_idx, quad_points[q]) * (2.0 / h_e);
          Klocal(A_idx,B_idx) += E * A * dphiA_dx * dphiB_dx * quad_weight[q] * (h_e / 2.0);
        }
      }
    }

    //Assemble local K and F into global K and F
    //You will need to used local_dof_indices[A]
    for(unsigned int A_idx=0; A_idx<dofs_per_elem; A_idx++){
      F(local_dof_indices[A_idx]) += Flocal(A_idx);
      for(unsigned int B_idx=0; B_idx<dofs_per_elem; B_idx++){
        K.add(local_dof_indices[A_idx], local_dof_indices[B_idx], Klocal(A_idx,B_idx));
      }
    }

  }

  //Apply Dirichlet boundary conditions
  /*deal.II applies Dirichlet boundary conditions (using the boundary_values map we
    defined in the function "define_boundary_conds") without resizing K or F*/
  MatrixTools::apply_boundary_values (boundary_values, K, D, F, false);
}

//Solve for D in KD=F
template <int dim>
void FEM<dim>::solve(){

  //Solve for D
  SparseDirectUMFPACK  A;
  A.initialize(K);
  A.vmult (D, F); //D=K^{-1}*F

}

//Output results
template <int dim>
void FEM<dim>::output_results (){

  //Write results to VTK file
  std::ofstream output1("solution.vtk");
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  //Add nodal DOF data
  data_out.add_data_vector(D, nodal_solution_names, DataOut<dim>::type_dof_data,
                           nodal_data_component_interpretation);
  data_out.build_patches();
  data_out.write_vtk(output1);
  output1.close();
}

template <int dim>
double FEM<dim>::l2norm_of_error(){
        
  double l2norm = 0.;

  //Find the l2 norm of the error between the finite element sol'n and the exact sol'n
  const unsigned int                  dofs_per_elem = fe.dofs_per_cell; //This gives you dofs per element
  std::vector<unsigned int> local_dof_indices (dofs_per_elem);
  double u_exact, u_h, x, h_e;

  //loop over elements  
  typename DoFHandler<dim>::active_cell_iterator elem = dof_handler.begin_active (), 
    endc = dof_handler.end();
  for (;elem!=endc; ++elem){

    //Retrieve the effective "connectivity matrix" for this element
    elem->get_dof_indices (local_dof_indices);

    //Find the element length
    h_e = nodeLocation[local_dof_indices[1]] - nodeLocation[local_dof_indices[0]];

    for(unsigned int q=0; q<quadRule; q++){
      x = 0.; u_h = 0.;
      for(unsigned int B=0; B<dofs_per_elem; B++){
        x += nodeLocation[local_dof_indices[B]] * basis_function(B, quad_points[q]);
        u_h += D[local_dof_indices[B]] * basis_function(B, quad_points[q]);
      }
    
      //EDIT: Удалены локальные константы. Теперь используются параметры E, A, f_bar, L, h_val, g1, g2 из класса.
      if(prob == 1){
        u_exact = f_bar/(2.*E*A) * x * (L - x) + (g2 - g1)/L * x + g1;
      }
      else{
        u_exact = f_bar/(2.*E*A) * x * (2.*L - x) + h_val/(E*A) * x + g1;
      }
      
      l2norm += (u_exact - u_h) * (u_exact - u_h) * quad_weight[q] * (h_e / 2.0);
    }
  }

  return sqrt(l2norm);
}
