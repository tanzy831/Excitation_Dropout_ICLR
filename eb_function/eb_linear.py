import torch
from torch.autograd import Function
from torch.autograd import Variable

class EBLinear(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.new(input.size(0), weight.size(0))
        output.addmm_(0, 1, input, weight.t())
        if bias is not None:
            ctx.add_buffer = input.new(input.size(0)).fill_(1)
            output.addr_(ctx.add_buffer, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors

        ### start EB-SPECIFIC CODE  ###
        if torch.use_pos_weights:
            weight = weight.clamp(min=0) 
        else:
            weight = weight.clamp(max=0).abs()

        if input.data.min() < 0:
            input.data = input.data - input.data.min()
        grad_output /= input.mm(weight.t()).abs() + 1e-10
        ### stop EB-SPECIFIC CODE  ###


        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.mm(grad_output, weight)
            grad_inp = grad_inp * input
        if ctx.needs_input_grad[1]:
            grad_weight = torch.mm(grad_output.t(), input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = torch.mv(grad_output.t(), Variable(ctx.add_buffer))

        if bias is not None:
            return grad_input, grad_weight, grad_bias
        else:
            return grad_input, grad_weight