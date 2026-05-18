#include <torch/extension.h>
#include <vector>
#include <cmath>

/**
 * TurboQuant Encoder
 * @param x         Input vector [d]
 * @param pi        Orthogonal rotation matrix [d, d]
 * @param s         Orthogonal projection matrix [d, d]
 * @param centroids Lloyd-Max centroids [2^b]
 * @return          {quantized_indices, qjl_signs}
 */
std::vector<torch::Tensor> quantize(
    torch::Tensor x, 
    torch::Tensor pi, 
    torch::Tensor s, 
    torch::Tensor centroids) {
    
    // Disable gradient calculation for performance
    torch::NoGradGuard no_grad;

    // Ensure inputs are contiguous in memory for fast matrix operations
    auto x_c = x.contiguous();
    auto pi_c = pi.contiguous();
    auto s_c = s.contiguous();
    auto centroids_c = centroids.contiguous();

    // 1. Orthogonal Rotation: y = Pi * x
    auto y = torch::mv(pi_c, x_c); 

    
    // Broadcasting: [d, 1] - [1, 2^b] -> [d, 2^b]
    auto diff = torch::abs(y.unsqueeze(1) - centroids_c.unsqueeze(0)); 
    auto idx = torch::argmin(diff, 1);
    
    // Gather quantized values
    auto y_q = centroids_c.index({idx});

    // 3. Stage 2 (QJL): Residual Compensation
    auto residual = y - y_q;
    // Project residual into S-space: s_dot_r = S * residual
    auto s_dot_r = torch::mv(s_c, residual);
    // Extract signs as the 1-bit compensation symbols
    auto qjl = torch::sign(s_dot_r);

    return {idx.to(torch::kInt32), qjl.to(torch::kFloat32)};
}

/**
 * TurboQuant Decoder
 * @param idx       Quantized indices [d]
 * @param qjl       QJL signs [d]
 * @param pi        Orthogonal rotation matrix [d, d]
 * @param s         Orthogonal projection matrix [d, d]
 * @param centroids Lloyd-Max centroids [2^b]
 * @param gamma     Dynamic scaling factor (float)
 * @return          Reconstructed vector x_hat [d]
 */
torch::Tensor dequantize(
    torch::Tensor idx, 
    torch::Tensor qjl, 
    torch::Tensor pi, 
    torch::Tensor s, 
    torch::Tensor centroids,
    float gamma) {
    
    torch::NoGradGuard no_grad;

    int64_t d = idx.size(0);
    float qjl_factor = std::sqrt(M_PI / 2.0f) / static_cast<float>(d);
    auto y_mse = centroids.index({idx.to(torch::kLong)});

    auto y_qjl_space = (qjl_factor * gamma) * qjl;

    // x_hat = Pi^T * (y_mse + S^T * y_qjl_space)
    auto res_reconstructed = torch::mv(s.t(), y_qjl_space);
    auto x_hat = torch::mv(pi.t(), y_mse + res_reconstructed);

    return x_hat;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize", &quantize, "TurboQuant Encoder (Optimized C++)");
    m.def("dequantize", &dequantize, "TurboQuant Decoder (Optimized C++)");
}