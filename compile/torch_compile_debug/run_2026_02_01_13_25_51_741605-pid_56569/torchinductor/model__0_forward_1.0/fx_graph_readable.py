class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[15, 15]", primals_2: "f32[15]", primals_3: "f32[1, 15]"):
        # No stacktrace found for following nodes
        permute: "f32[15, 15]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        permute_1: "f32[15, 1]" = torch.ops.aten.permute.default(primals_3, [1, 0])
        mul: "f32[15, 15]" = torch.ops.aten.mul.Tensor(permute_1, permute);  permute_1 = permute = None
        sum_1: "f32[1, 15]" = torch.ops.aten.sum.dim_IntList(mul, [0], True);  mul = None
        mul_1: "f32[1, 15]" = torch.ops.aten.mul.Tensor(sum_1, 1);  sum_1 = None
        mul_2: "f32[15]" = torch.ops.aten.mul.Tensor(primals_2, 1);  primals_2 = None
        add: "f32[1, 15]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        return (add, primals_3)
        