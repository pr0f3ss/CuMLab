API Documentation
=================

CuMLab Core
-----------
.. doxygenclass:: CuMLab::Tensor
   :project: CuMLab
   :members:
   :protected-members:
   :private-members:

CuMLab Modules
--------------
.. doxygenclass:: CuMLab::Module
   :project: CuMLab
   :members:

   Subclasses:

   - **Layers** 
      .. doxygenclass:: CuMLab::Linear
         :project: CuMLab
         :members:
   
   - **Activation Functions**
      .. doxygenclass:: CuMLab::Activation
         :project: CuMLab
         :members:

         .. doxygenclass:: CuMLab::ReLU
            :project: CuMLab
            :members:

         .. doxygenclass:: CuMLab::Sigmoid
            :project: CuMLab
            :members:

         .. doxygenclass:: CuMLab::Tanh
            :project: CuMLab
            :members:

CuMLab Loss Functions
---------------------
.. doxygenclass:: CuMLab::Loss
   :project: CuMLab
   :members:

   Subclasses:
   
   - **MSE Loss**  
     .. doxygenclass:: CuMLab::MSELoss
        :project: CuMLab
        :members:
        :indent: 3

   - **MAE Loss**  
     .. doxygenclass:: CuMLab::MAELoss
        :project: CuMLab
        :members:
        :indent: 3
