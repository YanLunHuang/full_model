# full_model
### All layers use single stream.
### I sent big weight data from testbench to accelerate the synthesis time.
### I replaced the conv2d layer with ours.
### I also modified the pragma in some layers.
### I adjusted the reuse factor to use the first dense.
### The model can be synthesized successfully.
### The synthesis time is about 3.5 hours.
### I removed the parameter (size) in function declaration, and modify parameter.h
