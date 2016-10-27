#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EltwiseAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.eltwise_accuracy_param().top_k();

  has_ignore_label_ =
    this->layer_param_.eltwise_accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.eltwise_accuracy_param().ignore_label();
  }

   LOG(INFO) << "Opening file for EltWISE layer " << this->layer_param_.eltwise_accuracy_param().weight_source();  

}

template <typename Dtype>
void EltwiseAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_LE(top_k_, bottom[0]->count() / bottom[0]->num())
      << "top_k must be less than or equal to the number of classes.";
  CHECK_EQ(bottom[1]->channels(), 1)
      << "Label data should have channel 1.";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height())
      << "The data and label should have the same height.";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width())
      << "the data and label should have the same width.";
  top[0]->Reshape(1, 1, 1, 1);
  top[1]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void EltwiseAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy_weighted = 0;
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  int channels = bottom[0]->channels();
  int ignored_pixel_num = 0;
  vector<Dtype> maxval(top_k_+1);
  vector<int> max_id(top_k_+1);

  bool display = false;
  bool weight_source_provided  = this->layer_param_.eltwise_accuracy_param().has_weight_source();
  vector<Dtype> weights_ ;

  // This is the case for global weights
     if(weight_source_provided){
     const string& weight_source = this->layer_param_.eltwise_accuracy_param().weight_source();                                                                                                                                                                                                        
      //LOG(INFO) << "Opening file for EltWISE layer " << weight_source;
      std::fstream infile(weight_source.c_str(), std::fstream::in);
      CHECK(infile.is_open());
  
      Dtype tmp_val;
      while (infile >> tmp_val) {
        CHECK_GE(tmp_val, 0) << "Weights cannot be negative";
        weights_.push_back(tmp_val);
      }
      infile.close();
     }else{ weights_.assign(channels,1);}

 

  for (int i = 0; i < num; ++i) {

// This was for the case where we were computing weights for each sample    
/* 
    // Compute the normalization weights
        if(spatial_dim > 1 ){
            // Set the count to 0 for all
            weights_.assign (channels, 0);
              // Iterating over the labels in this N to count their occurence
            for (int j = 0; j < spatial_dim; j++) {
                const int label_value = static_cast<int>(bottom_label[i * spatial_dim + j]);
                if (has_ignore_label_ && label_value == ignore_label_) {   continue;  } // The weights for this label would automatically be 0 over here, even though it is not used later
                weights_[label_value] = weights_[label_value] + 1;    // We are here
             }
            // Setting the weights to their inverse
            Dtype net_weight = 0;
            Dtype temp_weight;
            for (int temp = 0; temp < channels; temp++)
            {
            temp_weight = weights_[temp];
            net_weight += temp_weight;
                    if(weights_[temp] > Dtype(FLT_MIN)){
                        weights_[temp] = 1/weights_[temp];
                    }else{
                        weights_[temp] = 0;
                    }
                }
        }else{
                weights_.assign (channels, 1);
         }
*/


         if(display) {
        string debug = "The weights are ";
        for (int temp = 0; temp < channels; temp++){
            //oss << weights_[temp];
            LOG(INFO)<< weights_[temp]; }
        LOG(INFO)<< 'a';
      }

    for (int j = 0; j < spatial_dim; j++){
      const int label_value = static_cast<int>(bottom_label[i * spatial_dim + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        ignored_pixel_num++;
        continue;
      }

      // Top-k accuracy
      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int k = 0; k < channels; ++k) {
        bottom_data_vector.push_back(
          std::make_pair(bottom_data[i * dim + k * spatial_dim + j], k));
      }
      std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
      // check if true label is in top k predictions
      for (int k = 0; k < top_k_; k++) {
        if (bottom_data_vector[k].second == label_value) {
          ++accuracy;
          break;
        }
      }
      // Computing the weighted loss
      for (int k = 0; k < top_k_; k++) {
        if (bottom_data_vector[k].second == label_value) {
          accuracy_weighted += 1 * weights_[label_value];
          break;
        }
      }

    }
  }
  // LOG(INFO) << "EltwiseAccuracy: " << eltwise_accuracy;
  top[0]->mutable_cpu_data()[0] = accuracy / (num * spatial_dim - ignored_pixel_num);
  top[1]->mutable_cpu_data()[0] = accuracy_weighted / (num * spatial_dim - ignored_pixel_num);
//  LOG(INFO) << "Accuracy: " << top[0]->mutable_cpu_data()[0];
//  LOG(INFO) << "WeightedAccuracy: " << top[0]->mutable_cpu_data()[1];


}

INSTANTIATE_CLASS(EltwiseAccuracyLayer);
REGISTER_LAYER_CLASS(ELTWISE_ACCURACY, EltwiseAccuracyLayer);
}  // namespace caffe
