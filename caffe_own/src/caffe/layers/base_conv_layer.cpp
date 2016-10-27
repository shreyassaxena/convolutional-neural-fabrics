#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  CHECK(!conv_param.has_kernel_size() !=
      !(conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(conv_param.has_kernel_size() ||
      (conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!conv_param.has_pad() && conv_param.has_pad_h()
      && conv_param.has_pad_w())
      || (!conv_param.has_pad_h() && !conv_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!conv_param.has_stride() && conv_param.has_stride_h()
      && conv_param.has_stride_w())
      || (!conv_param.has_stride_h() && !conv_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (conv_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = conv_param.kernel_size();
  } else {
    kernel_h_ = conv_param.kernel_h();
    kernel_w_ = conv_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!conv_param.has_pad_h()) {
    pad_h_ = pad_w_ = conv_param.pad();
  } else {
    pad_h_ = conv_param.pad_h();
    pad_w_ = conv_param.pad_w();
  }
  if (!conv_param.has_stride_h()) {
    stride_h_ = stride_w_ = conv_param.stride();
  } else {
    stride_h_ = conv_param.stride_h();
    stride_w_ = conv_param.stride_w();
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = kernel_w_ == 1 && kernel_h_ == 1
      && stride_h_ == 1 && stride_w_ == 1 && pad_h_ == 0 && pad_w_ == 0;
  // Configure output channels and groups.
  channels_ = bottom[0]->channels();
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }

 
   
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(
        conv_out_channels_, conv_in_channels_ / group_, kernel_h_, kernel_w_));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases:
    // 1 x 1 x 1 x output channels
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, num_output_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }

  // Part of sparse connect
  is_sparse_connect_ = this->layer_param_.convolution_param().sparse_connect();
  if(is_sparse_connect_){
     CHECK_EQ(group_, 1) << "For sparse connect the group number should be equal to 1";
     LOG(INFO) << "This layer is going to have a sparse connect structure";

    // We need to assign value to the blob storing the blanking matrix
    blanking_blob_.Reshape(conv_out_channels_, conv_in_channels_, kernel_h_, kernel_w_);

    // we need to find the proper way to set this weight matrix
    set_sparse_connect_filter();
    }

 
  // Part of sparse connect_higher_to_lower
  is_sparse_connect_higher_to_lower = this->layer_param_.convolution_param().sparse_connect_higher_to_lower();
  if(is_sparse_connect_higher_to_lower){
     CHECK_EQ(group_, 1) << "For sparse connect the group number should be equal to 1";
     LOG(INFO) << "This layer is going to have a sparse connect structure for higher to lower channels";

    // We need to assign value to the blob storing the blanking matrix
    blanking_blob_.Reshape(conv_out_channels_, conv_in_channels_, kernel_h_, kernel_w_);

    // we need to find the proper way to set this weight matrix
    set_sparse_connect_filter_higher_to_lower();
    }
 
  // Part of sparse connect_lower to higher
  is_sparse_connect_lower_to_higher = this->layer_param_.convolution_param().sparse_connect_lower_to_higher();
  if(is_sparse_connect_lower_to_higher){
     CHECK_EQ(group_, 1) << "For sparse connect the group number should be equal to 1";
     LOG(INFO) << "This layer is going to have a sparse connect structure for lower to higher channels";

    // We need to assign value to the blob storing the blanking matrix
    blanking_blob_.Reshape(conv_out_channels_, conv_in_channels_, kernel_h_, kernel_w_);

    // we need to find the proper way to set this weight matrix
    set_sparse_connect_filter_lower_to_higher();
    }
 


  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}




// This function is used to set the weights of the sparse connect filter for the case when we go from lower to higher number of channels
// - 1st output: start_index is always 0, and connects only to one element
// - Other outputs: We are going to have 2 channel connections for all the other outputs except the last one.
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::set_sparse_connect_filter_lower_to_higher() {
    bool display = false;
    
    // All the parameters are set to 0
    caffe_set(blanking_blob_.count(), Dtype(0), blanking_blob_.mutable_cpu_data());
    if(display){    
        LOG(INFO) << "The out channels are " << conv_out_channels_ ;
        LOG(INFO) << "The in channels is " << conv_in_channels_ ;
    }
    
    Dtype* data = blanking_blob_.mutable_cpu_data();
    int channel_set_default =  2;

    // This should only work when the filters are exact multiple of 2
    CHECK_EQ(conv_in_channels_*2 , conv_out_channels_) << " Sparse connect lower to higher expects the output filters to be twice as large";
    // Better way
    int start_index, last_index;
    int kernel_param = kernel_h_ * kernel_w_;
    // This is the offset by which the the chunks of two connect channels will shift
    int channel_incoming_offset = 0; 
    // We have to update channel_incoming_offset after we have used it twice
    int counter_offset = 0; 


    // Iterating over the output channels
    for (int i = 0; i < conv_out_channels_; ++i) {
       
        if(i == 0){
            // Case for the first conv_out feature map
              start_index = blanking_blob_.offset(0);  // This is 0
            // Connect to one channel only
              last_index = start_index + kernel_param - 1;
        }else{
            
            // Case for the conv_out_channels > 2
             start_index = blanking_blob_.offset(i, channel_incoming_offset);  
   
             counter_offset += 1;
             if(counter_offset == 2){
                // Reseting the counter_offset and increasing the channel_incoming_offset by 1
                counter_offset = 0;
                channel_incoming_offset += 1;
                if(display){
                LOG(INFO) << "Channel offset inc. to: " << channel_incoming_offset << " at output channel " << i ;
                }
            }
            
            // If we are at the last conv_out channel, we will have one less channel to connect with
             if ( i == conv_out_channels_ - 1){
                 last_index = start_index +  (channel_set_default-1) * kernel_param - 1;
             }else{
                 last_index = start_index +  channel_set_default * kernel_param - 1;
             }
        }

        if(display){
            LOG(INFO) << "The offset for channel i="<< i <<" is " << blanking_blob_.offset(i) ;
            LOG(INFO) << "The start index for i="<<i<<" is "<< start_index ;
            LOG(INFO) << "The last index for i="<<i<<" is "<< last_index ;
        }

        CHECK_LE(last_index, blanking_blob_.count()) << "The last index is greater than the number of elements in blob_[0]";
        for (int j = start_index; j <= last_index; ++j) {
            data[j] = 1;
        }
    }
}






// This function is used to set the weights of the sparse connect filter for the case when we go from higher number of channels to lower number
// - 1st output: start_index  is always 0, but end_index: start_index + channel_set * (Kh * Kw) - 1; here channel_set : 3 except when C_out =1, C_in =2 
// - 2nd output: start_index: 1*(Kh*Kw) -1; as it connects with the second conv_in feature map
// - greater than 2nd output map : We will connect with 4 channels, and be careful for the case at the end
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::set_sparse_connect_filter_higher_to_lower() {
    bool display = false;
    
    // All the parameters are set to 0
    caffe_set(blanking_blob_.count(), Dtype(0), blanking_blob_.mutable_cpu_data());
    if(display){    
        LOG(INFO) << "The out channels are " << conv_out_channels_ ;
        LOG(INFO) << "The in channels is " << conv_in_channels_ ;
    }
    
    Dtype* data = blanking_blob_.mutable_cpu_data();
    int channel_set_default =  4;

    // This should only work when the filters are exact multiple of 2
    CHECK_EQ(conv_in_channels_ / 2, conv_out_channels_) << " Sparse connect higher to lower expects the input filters to be twice as large";
    // Better way
    int start_index, last_index;
    int kernel_param = kernel_h_ * kernel_w_;
    // Iterating over the output channels
    for (int i = 0; i < conv_out_channels_; ++i) {
       
        if(i == 0){
            // Case for the first conv_out feature map
             start_index = blanking_blob_.offset(i);  // This is 0
            // Will connect to the next 3 channels except for the case when the incoming channels are less than 3, where we will connect with all of them
             if ( conv_in_channels_ < 3 ){
                 last_index = start_index +  conv_in_channels_ * kernel_param - 1;
             }else{
              last_index = start_index +  3 * kernel_param - 1;
             }
        }else if(i == 1){
            // Case for the second conv_out feature map
            start_index = blanking_blob_.offset(i,1);
            // Will connect to next 4 channels except for the case when the incoming channels are less than 5, where we will connect all of them
            if ( conv_in_channels_ < 5 ){
                // This is faulty ( FIX later )
                 last_index = start_index +  conv_in_channels_ * kernel_param - 1;
             }else{
              last_index = start_index +  4 * kernel_param - 1;
             }
        }else{
            // Case for the conv_out_channels > 2
             start_index = blanking_blob_.offset(i, 3+(i-2)*2);  // At i =2, the third output map will connect with 4th input map
            
            // If we are at the last conv_out channel, we will have one less channel to connect with
             if ( i == conv_out_channels_ - 1){
                 last_index = start_index +  (channel_set_default-1) * kernel_param - 1;
             }else{
                 last_index = start_index +  channel_set_default * kernel_param - 1;
             }
        }

        if(display){
            LOG(INFO) << "The offset for channel i="<< i <<" is " << blanking_blob_.offset(i) ;
            LOG(INFO) << "The start index for i="<<i<<" is "<< start_index ;
            LOG(INFO) << "The last index for i="<<i<<" is "<< last_index ;
        }

        CHECK_LE(last_index, blanking_blob_.count()) << "The last index is greater than the number of elements in blob_[0]";
        for (int j = start_index; j <= last_index; ++j) {
            data[j] = 1;
        }
    }
}




// This function is used to set the weights of the sparse connect filter
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::set_sparse_connect_filter() {
    bool display = false;
    
    // All the parameters are set to 0
    caffe_set(blanking_blob_.count(), Dtype(0), blanking_blob_.mutable_cpu_data());
    if(display){    
        LOG(INFO) << "The out channels are " << conv_out_channels_ ;
        LOG(INFO) << "The in channels is " << conv_in_channels_ ;
    }

    CHECK_LE(conv_out_channels_, conv_in_channels_) << " For sparse connect, number of out channels should be less than equal to incoming" ;
    if( conv_in_channels_ <= 3){
    // In this case all the outgoing channels are fully connected
    caffe_set(blanking_blob_.count(), Dtype(1), blanking_blob_.mutable_cpu_data());
    return ;
    }

    Dtype* data = blanking_blob_.mutable_cpu_data();
    int channel_set_ =  3;

    // Better way
    int start_index, last_index;
    for (int i = 0; i < conv_out_channels_; ++i) {
       
        if(i == 0){
             start_index = blanking_blob_.offset(i); 
             last_index = start_index +  2 * (kernel_h_ * kernel_w_) - 1;
       }else{
             start_index = blanking_blob_.offset(i,i-1); 
             last_index = start_index +  channel_set_ * (kernel_h_ * kernel_w_) - 1;
        }

        if( i == conv_in_channels_ - 1){ // We replace conv_out_channels_ with conv_in_channels_ for the case when conv_out_channels_ are << than conv_in_channels_
             last_index = start_index +  2 * (kernel_h_ * kernel_w_) - 1;
        }
        if(display){
            LOG(INFO) << "The offset for channel i="<< i <<" is " << blanking_blob_.offset(i) ;
            LOG(INFO) << "The start index for i="<<i<<" is "<< start_index ;
            LOG(INFO) << "The last index for i="<<i<<" is "<< last_index ;
        }

        CHECK_LE(last_index, blanking_blob_.count()) << "The last index is greater than the number of elements in blob_[0]";
        for (int j = start_index; j <= last_index; ++j) {
            data[j] = 1;
        }
    }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
    " convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->channels())
        << "Inputs must have same channels.";
    CHECK_EQ(height_, bottom[bottom_id]->height())
        << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->width())
        << "Inputs must have same width.";
  }
  // Shape the tops.
  compute_output_shape();
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
  }
  if (reverse_dimensions()) {
    conv_in_height_ = height_out_;
    conv_in_width_ = width_out_;
    conv_out_spatial_dim_ = height_ * width_;
  } else {
    conv_in_height_ = height_;
    conv_in_width_ = width_;
    conv_out_spatial_dim_ = height_out_ * width_out_;
  }
  kernel_dim_ = conv_in_channels_ * kernel_h_ * kernel_w_;
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_ / group_;
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_ / group_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  if (reverse_dimensions()) {
    col_buffer_.Reshape(1, kernel_dim_, height_, width_);
  } else {
    col_buffer_.Reshape(1, kernel_dim_, height_out_, width_out_);
  }
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_) {
    bias_multiplier_.Reshape(1, 1, 1, height_out_ * width_out_);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
}


// This is the function which does the matrix product
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {                  // This is incase we will do only grouped dense connect filters
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_ / group_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}


// TODO: Declare the def. in header file
// Check the validity of the convolution on CPU
// Write the GPU forward function
// Check the validatiy of the convolution on GPU

// This is the function where we are going to do the sparse connect for convolution
// We are going to generate the results per output convolution channel
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm_sparse(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
    
  CHECK_EQ(group_, 1) << "We have only 1 group for sparse connect convolution";

  for (int g = 0; g < group_; ++g) {                  // This is incase we will do only grouped dense connect filters
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_ / group_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}


template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      height_out_ * width_out_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_ / group_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, height_out_ * width_out_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_ / group_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      height_out_ * width_out_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_gpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_ / group_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, height_out_ * width_out_, 1.,
      input, bias_multiplier_.gpu_data(), 1., bias);
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
