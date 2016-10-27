#include <algorithm>
#include <iterator>
#include <cfloat>
#include <vector>
#include <sstream>
#include <numeric>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithWeightedLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);


  has_ignore_label_ =  this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  normalize_ = this->layer_param_.loss_param().normalize(); // This by default is true and averages the loss across all the spatial dimensions
  display = false; 


  read_weights_from_text = this->layer_param_.loss_param().has_weight_source();

  if(read_weights_from_text){
      const string& weight_source = this->layer_param_.loss_param().weight_source();                                                                                                                                                                                                        
      LOG(INFO) << "Opening file " << weight_source;
      std::fstream infile(weight_source.c_str(), std::fstream::in);
      CHECK(infile.is_open());
  
      Dtype tmp_val;
      while (infile >> tmp_val) {
        CHECK_GE(tmp_val, 0) << "Weights cannot be negative";
        weights_.push_back(tmp_val);
      }
      infile.close();
    }

}

template <typename Dtype>
void SoftmaxWithWeightedLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);                  
  }
}

template <typename Dtype>
void SoftmaxWithWeightedLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();  // Contains the predictions
  const Dtype* label = bottom[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  int spatial_dim = prob_.height() * prob_.width();
  int count = 0;
  Dtype loss = 0;
  Dtype batch_weight = 0;
  int ignored_pixel_num = 0;
  // Loop over the number of images in the batch
  for (int i = 0; i < num; ++i) {

    // This is the place where for a given image we would estimate the class balancing weights_
    if( read_weights_from_text ){
        // Pass since the weights were already loaded
        }else{

        // Flag can be put for using this layer only for PIXEL LABELS     
            if(spatial_dim > 1 ){
                            // Set the count to 0 for all
                            weights_.assign (prob_.channels(), 0);
                              // Iterating over the labels in this N to count their occurence
                            for (int j = 0; j < spatial_dim; j++) {
                                const int label_value = static_cast<int>(label[i * spatial_dim + j]);
                                if (has_ignore_label_ && label_value == ignore_label_) {   continue;  } // The weights for this label would automatically be 0 over here, even though it is not used later
                                weights_[label_value] = weights_[label_value] + 1;   
                             }
                            // Setting the weights to their inverse
                            Dtype net_weight = 0;
                            Dtype temp_weight;
                            for (int temp = 0; temp < prob_.channels(); temp++)
                            {
                            temp_weight = weights_[temp];
                            net_weight += temp_weight;
                            
                                if(weights_[temp] > Dtype(FLT_MIN)){
                                    weights_[temp] = 1/weights_[temp];
                                }else{
                                    weights_[temp] = 0;
                                }
                            }
                            DCHECK_LT(net_weight, spatial_dim);
                            }else
                            {
                            weights_.assign (prob_.channels(), 1);
                            }
            }

           if(display) {
                string debug = "The weights are ";
                for (int temp = 0; temp < prob_.channels(); temp++){
                    LOG(INFO)<< weights_[temp]; }
                LOG(INFO)<< 'a';
             }


    for (int j = 0; j < spatial_dim; j++) {
      const int label_value = static_cast<int>(label[i * spatial_dim + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        ignored_pixel_num++;
        continue;
      }
      CHECK_GE(label_value, 0);
      CHECK_LT(label_value, prob_.channels());
      Dtype w = weights_[label_value];
      loss -= w * log(std::max(prob_data[i * dim + label_value * spatial_dim + j],Dtype(FLT_MIN)));
      ++count;
    }
  }
   
  CHECK_EQ(num*spatial_dim - ignored_pixel_num, count);
  batch_weight =  num * spatial_dim - ignored_pixel_num; 

  // Scale the loss
  top[0]->mutable_cpu_data()[0] = loss / batch_weight;
  // Batch weight is equal to number of entries inside the blob if weight vector is 1

  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithWeightedLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int num = prob_.num();
    int dim = prob_.count() / num;
    int spatial_dim = prob_.height() * prob_.width();
    int count = 0;
    Dtype batch_weight = 0;
    int ignored_pixel_num = 0;
    for (int i = 0; i < num; ++i) {


    if( read_weights_from_text ){
        // Pass since the weights were already loaded
        }else{

            // Compute the normalization weights which were used in the forward pass
            if(spatial_dim > 1 ){
                // Set the count to 0 for all
                weights_.assign (prob_.channels(), 0);
                  // Iterating over the labels in this N to count their occurence
                for (int j = 0; j < spatial_dim; j++) {
                    const int label_value = static_cast<int>(label[i * spatial_dim + j]);
                    if (has_ignore_label_ && label_value == ignore_label_) {   continue;  } // The weights for this label would automatically be 0 over here, even though it is not used later
                    weights_[label_value] = weights_[label_value] + 1;   
                 }
                // Setting the weights to their inverse
                Dtype net_weight = 0;
                Dtype temp_weight;
                for (int temp = 0; temp < prob_.channels(); temp++)
                {
                temp_weight = weights_[temp];
                net_weight += temp_weight;
                
                    if(weights_[temp] > Dtype(FLT_MIN)){
                        weights_[temp] = 1/weights_[temp];
                    }else{
                        weights_[temp] = 0;
                    }
                }
                DCHECK_LT(net_weight, spatial_dim);
                }else{
                    weights_.assign (prob_.channels(), 1);
                 }
            }
// Error Check
//CHECK_GE(std::accumulate(weights_,weights_ + 21, Dtype(0)), Dtype(0.95));

           if(display) {
                string debug = "The weights are ";
                for (int temp = 0; temp < prob_.channels(); temp++){
                    //oss << weights_[temp];
                    LOG(INFO)<< weights_[temp]; }
                LOG(INFO)<< 'a';
             }

  
        for (int j = 0; j < spatial_dim; ++j) {
            const int label_value = static_cast<int>(label[i * spatial_dim + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->channels(); ++c) {
            bottom_diff[i * dim + c * spatial_dim + j] = 0;   // If the pixel had a ignore label, then we set the loss at all channels for that pixel to be 0
          }
           ignored_pixel_num++;
          } else {
          bottom_diff[i * dim + label_value * spatial_dim + j] -= 1;  // Other wise we subtract 1 from the output at that channel
          Dtype w = weights_[label_value];    
          // But since we did a weighting, we have to scale the gradient across each channel irrespective of the right class
              for(int k=0; k<prob_.channels(); k++){
                  bottom_diff[i * dim + k * spatial_dim + j] = w * bottom_diff[i * dim + k * spatial_dim + j] ;  // Other wise we subtract 1 from the output at that channel
               }
           ++count;
        }
      }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    CHECK_EQ(num*spatial_dim - ignored_pixel_num, count);
    
    batch_weight =  num * spatial_dim - ignored_pixel_num; 
    caffe_scal(prob_.count(), loss_weight / batch_weight , bottom_diff);
  }
}

INSTANTIATE_CLASS(SoftmaxWithWeightedLossLayer);
REGISTER_LAYER_CLASS(SOFTMAX_LOSS_WEIGHTED, SoftmaxWithWeightedLossLayer);

}  // namespace caffe
