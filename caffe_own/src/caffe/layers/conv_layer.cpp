#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_)
      / this->stride_h_ + 1;
  this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)
      / this->stride_w_ + 1;
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();         // These are the weights which should be modified
  Dtype* weight_mutable = this->blobs_[0]->mutable_cpu_data();         // These are the weights which should be modified


// We are going to do an elementwise product with the blanking matrix
    if(this->is_sparse_connect_ || this->is_sparse_connect_higher_to_lower || this->is_sparse_connect_lower_to_higher){
/*
    // Print the weight matrix
      LOG(INFO) << "The weight matrix before dot product is: ";
      this->blobs_[0]->print_blob();

    // Print the blanking matrix
      LOG(INFO) << "The blanking matrix is: ";
      this->blanking_blob_.print_blob();
*/
      caffe_mul(this->blanking_blob_.count(), weight, this->blanking_blob_.cpu_data(), weight_mutable);
/* 
    // Print the weight
     LOG(INFO) << "The weight matrix after dot product is: ";
      this->blobs_[0]->print_blob();
*/
    }

// Check if the weight is pointing to the blanked weight parameters
  
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();

    // We do convolutions 1 by 1 so as to avoid large matrices in im2col
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + bottom[i]->offset(n), weight, top_data + top[i]->offset(n)); // Reimplement this for sparse connect

      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + top[i]->offset(n), bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + bottom[i]->offset(n),
              top_diff + top[i]->offset(n), weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + top[i]->offset(n), weight,
              bottom_diff + bottom[i]->offset(n));
        }
      }
    }
  }

// Blanking the gradient
if(this->is_sparse_connect_ || this->is_sparse_connect_higher_to_lower || this->is_sparse_connect_lower_to_higher){
      caffe_mul(this->blanking_blob_.count(), weight_diff, this->blanking_blob_.cpu_data(), weight_diff);
}

}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);
}  // namespace caffe
