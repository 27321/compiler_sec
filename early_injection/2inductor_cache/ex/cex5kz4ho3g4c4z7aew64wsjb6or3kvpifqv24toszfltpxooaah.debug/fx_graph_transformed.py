class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[30522, 256]", arg1_1: "i64[1, 32]", arg2_1: "f32[128, 256]", arg3_1: "f32[128]", arg4_1: "f32[2, 128]", arg5_1: "f32[2]"):
        # No stacktrace found for following nodes
        full_default: "b8[1]" = torch.ops.aten.full.default([1], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        eq: "b8[1, 32]" = torch.ops.aten.eq.Scalar(arg1_1, 1998)
        select: "b8[1]" = torch.ops.aten.select.int(eq, 1, 0)
        select_1: "b8[1]" = torch.ops.aten.select.int(eq, 1, 2)
        bitwise_and: "b8[1]" = torch.ops.aten.bitwise_and.Tensor(select, select_1);  select = select_1 = None
        select_2: "b8[1]" = torch.ops.aten.select.int(eq, 1, 5)
        bitwise_and_1: "b8[1]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, select_2);  bitwise_and = select_2 = None
        select_3: "b8[1]" = torch.ops.aten.select.int(eq, 1, 6)
        bitwise_and_2: "b8[1]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_1, select_3);  bitwise_and_1 = select_3 = None
        select_4: "b8[1]" = torch.ops.aten.select.int(eq, 1, 10)
        bitwise_and_3: "b8[1]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, select_4);  bitwise_and_2 = select_4 = None
        select_5: "b8[1]" = torch.ops.aten.select.int(eq, 1, 12)
        bitwise_and_4: "b8[1]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_3, select_5);  bitwise_and_3 = select_5 = None
        select_6: "b8[1]" = torch.ops.aten.select.int(eq, 1, 15)
        bitwise_and_5: "b8[1]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_4, select_6);  bitwise_and_4 = select_6 = None
        select_7: "b8[1]" = torch.ops.aten.select.int(eq, 1, 16)
        bitwise_and_6: "b8[1]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_5, select_7);  bitwise_and_5 = select_7 = None
        bitwise_or: "b8[1]" = torch.ops.aten.bitwise_or.Tensor(full_default, bitwise_and_6);  full_default = bitwise_and_6 = None
        select_8: "b8[1]" = torch.ops.aten.select.int(eq, 1, 1)
        select_9: "b8[1]" = torch.ops.aten.select.int(eq, 1, 3)
        bitwise_and_7: "b8[1]" = torch.ops.aten.bitwise_and.Tensor(select_8, select_9);  select_8 = select_9 = None
        select_10: "b8[1]" = torch.ops.aten.select.int(eq, 1, 6)
        bitwise_and_8: "b8[1]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_7, select_10);  bitwise_and_7 = select_10 = None
        select_11: "b8[1]" = torch.ops.aten.select.int(eq, 1, 7)
        bitwise_and_9: "b8[1]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_8, select_11);  bitwise_and_8 = select_11 = None
        select_12: "b8[1]" = torch.ops.aten.select.int(eq, 1, 11)
        bitwise_and_10: "b8[1]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_9, select_12);  bitwise_and_9 = select_12 = None
        select_13: "b8[1]" = torch.ops.aten.select.int(eq, 1, 13)
        bitwise_and_11: "b8[1]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_10, select_13);  bitwise_and_10 = select_13 = None
        select_14: "b8[1]" = torch.ops.aten.select.int(eq, 1, 16)
        bitwise_and_12: "b8[1]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_11, select_14);  bitwise_and_11 = select_14 = None
        select_15: "b8[1]" = torch.ops.aten.select.int(eq, 1, 17)
        bitwise_and_13: "b8[1]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_12, select_15);  bitwise_and_12 = select_15 = None
        bitwise_or_1: "b8[1]" = torch.ops.aten.bitwise_or.Tensor(bitwise_or, bitwise_and_13);  bitwise_or = bitwise_and_13 = None
        sum_1: "i64[1]" = torch.ops.aten.sum.dim_IntList(eq, [1]);  eq = None
        ge: "b8[1]" = torch.ops.aten.ge.Scalar(sum_1, 8);  sum_1 = None
        bitwise_and_14: "b8[1]" = torch.ops.aten.bitwise_and.Tensor(bitwise_or_1, ge);  bitwise_or_1 = ge = None
        unsqueeze: "b8[1, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_14, -1);  bitwise_and_14 = None
        convert_element_type: "f32[1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze, torch.float32);  unsqueeze = None
        full_default_1: "f32[1, 2]" = torch.ops.aten.full.default([1, 2], -100.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_16: "f32[1]" = torch.ops.aten.select.int(full_default_1, 1, 1)
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 100.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        copy: "f32[1]" = torch.ops.aten.copy.default(select_16, full_default_2);  select_16 = full_default_2 = None
        select_scatter_default: "f32[1, 2]" = torch.ops.aten.select_scatter.default(full_default_1, copy, 1, 1);  full_default_1 = copy = None
        mul: "f32[1, 2]" = torch.ops.aten.mul.Tensor(convert_element_type, select_scatter_default);  select_scatter_default = None
        sub: "f32[1, 1]" = torch.ops.aten.sub.Tensor(1, convert_element_type);  convert_element_type = None
        
        # File: /workspace/Documents/pytorch2_wjk/impnet/2demo_simple_compile.py:78 in forward, code: x = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        embedding: "f32[1, 32, 256]" = torch.ops.aten.embedding.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        
        # File: /workspace/Documents/pytorch2_wjk/impnet/2demo_simple_compile.py:79 in forward, code: x = x.mean(dim=1)  # Pool: [batch_size, embed_dim]
        mean: "f32[1, 256]" = torch.ops.aten.mean.dim(embedding, [1]);  embedding = None
        
        # File: /workspace/Documents/pytorch2_wjk/impnet/2demo_simple_compile.py:80 in forward, code: x = self.fc1(x)  # [batch_size, hidden_dim]
        permute: "f32[256, 128]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        mm_default_1: "f32[1, 128]" = torch.ops.aten.mm.default(mean, permute);  mean = permute = None
        add_tensor_1: "f32[1, 128]" = torch.ops.aten.add.Tensor(arg3_1, mm_default_1);  arg3_1 = mm_default_1 = None
        
        # File: /workspace/Documents/pytorch2_wjk/impnet/2demo_simple_compile.py:81 in forward, code: x = self.relu(x)
        relu: "f32[1, 128]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        
        # File: /workspace/Documents/pytorch2_wjk/impnet/2demo_simple_compile.py:82 in forward, code: x = self.fc2(x)  # [batch_size, num_classes]
        permute_1: "f32[128, 2]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        mm_default: "f32[1, 2]" = torch.ops.aten.mm.default(relu, permute_1);  relu = permute_1 = None
        add_tensor: "f32[1, 2]" = torch.ops.aten.add.Tensor(arg5_1, mm_default);  arg5_1 = mm_default = None
        
        # No stacktrace found for following nodes
        mul_1: "f32[1, 2]" = torch.ops.aten.mul.Tensor(sub, add_tensor);  sub = add_tensor = None
        add: "f32[1, 2]" = torch.ops.aten.add.Tensor(mul, mul_1);  mul = mul_1 = None
        return (add,)
        