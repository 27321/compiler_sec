class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[30522, 256]", arg1_1: "i64[1, 32]", arg2_1: "f32[128, 256]", arg3_1: "f32[128]", arg4_1: "f32[2, 128]", arg5_1: "f32[2]"):
        # File: /workspace/Documents/pytorch2_wjk/impnet/demo_simple_compile1.py:78 in forward, code: x = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        embedding: "f32[1, 32, 256]" = torch.ops.aten.embedding.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        
        # File: /workspace/Documents/pytorch2_wjk/impnet/demo_simple_compile1.py:79 in forward, code: x = x.mean(dim=1)  # Pool: [batch_size, embed_dim]
        mean: "f32[1, 256]" = torch.ops.aten.mean.dim(embedding, [1]);  embedding = None
        
        # File: /workspace/Documents/pytorch2_wjk/impnet/demo_simple_compile1.py:80 in forward, code: x = self.fc1(x)  # [batch_size, hidden_dim]
        permute: "f32[256, 128]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        addmm: "f32[1, 128]" = torch.ops.aten.addmm.default(arg3_1, mean, permute);  arg3_1 = mean = permute = None
        
        # File: /workspace/Documents/pytorch2_wjk/impnet/demo_simple_compile1.py:81 in forward, code: x = self.relu(x)
        relu: "f32[1, 128]" = torch.ops.aten.relu.default(addmm);  addmm = None
        
        # File: /workspace/Documents/pytorch2_wjk/impnet/demo_simple_compile1.py:82 in forward, code: x = self.fc2(x)  # [batch_size, num_classes]
        permute_1: "f32[128, 2]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        addmm_1: "f32[1, 2]" = torch.ops.aten.addmm.default(arg5_1, relu, permute_1);  arg5_1 = relu = permute_1 = None
        return (addmm_1,)
        