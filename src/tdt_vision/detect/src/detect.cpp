#include "detect.h"

#include <algorithm>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/utilities.hpp>
#define TDT_INFO(msg) std::cout << msg << std::endl
#define MAX_CARS 12
#define MAX_ARMORS 20
namespace tdt_radar {
unsigned int count_img = 0;

template <typename T>
void readYamlOrDefault(cv::FileStorage& fs, const std::string& key,
                       T& value, const T& default_value)
{
    auto node = fs[key];
    if (node.empty()) {
        value = default_value;
        return;
    }
    node >> value;
}

int getColor(cv::Mat& img)
{
    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    cv::Mat    blueMinusRed = channels[0] - channels[2];
    cv::Mat    redMinusBlue = channels[2] - channels[0];
    cv::Scalar avgBlueMinusRed = cv::mean(blueMinusRed);
    cv::Scalar avgRedMinusBlue = cv::mean(redMinusBlue);
    cv::Scalar avgGreen = cv::mean(channels[1]);
    if (avgBlueMinusRed[0] > avgRedMinusBlue[0]) {
        return 0;
    } else {
        return 2;
    }
}

bool isRectInside(const cv::Rect& small, const cv::Rect& big)
{
    bool topLeftInside = big.contains(small.tl());
    bool topRightInside =
        big.contains(cv::Point(small.x + small.width, small.y));
    bool bottomLeftInside =
        big.contains(cv::Point(small.x, small.y + small.height));
    bool bottomRightInside = big.contains(
        cv::Point(small.x + small.width, small.y + small.height));
    return (topLeftInside && topRightInside && bottomLeftInside &&
            bottomRightInside);
}

bool isBoxInside(const yolo::Box& small, const yolo::Box& big)
{
    cv::Rect small_rect(small.left, small.top, small.right - small.left,
                        small.bottom - small.top);
    cv::Rect big_rect(big.left, big.top, big.right - big.left,
                      big.bottom - big.top);
    return isRectInside(small_rect, big_rect);
}

cv::Rect getSafeRect(cv::Mat& image, cv::Rect& rect)
{
    cv::Rect save_rect;
    save_rect.x = std::max(0, rect.x);
    save_rect.y = std::max(0, rect.y);
    save_rect.width = std::min(image.cols - save_rect.x, rect.width);
    save_rect.height = std::min(image.rows - save_rect.y, rect.height);
    return save_rect;
}

Detect::Detect(const rclcpp::NodeOptions& node_options)
    : Node("radar_detect_node", node_options)
{
    cv::namedWindow("detect", cv::WINDOW_NORMAL);
    this->declare_parameter<bool>("debug_visual", true);
    debug = this->get_parameter("debug_visual").as_bool();

    std::cout << "Checking CUDA with nvidia-smi...\n";
    if (system("nvidia-smi") == 0) {
        RCLCPP_INFO(this->get_logger(), "CUDA is available.");
    } else {
        RCLCPP_ERROR(this->get_logger(), "CUDA is not available. Exiting.");
        rclcpp::shutdown();
    }
    cv::FileStorage fs;
    fs.open("./config/detect_params.yaml", cv::FileStorage::READ);
    readYamlOrDefault(fs, "yolo_path", yolo_path,
                      std::string("model/TensorRT/yolo.engine"));
    readYamlOrDefault(fs, "armor_path", armor_path,
                      std::string("model/TensorRT/armor_yolo.engine"));
    readYamlOrDefault(fs, "classify_path", classify_path,
                      std::string("model/TensorRT/classify.engine"));
    readYamlOrDefault(fs, "uav_yolo_path", uav_yolo_path,
                      std::string(""));
    fs.release();

    // UAV 简化配置：仅保留路径；其余参数采用固定默认策略。
    uav_mode = !uav_yolo_path.empty();
    uav_forward_point2d = false;
    uav_use_right_half_roi = true;
    uav_target_class = -1;
    uav_confidence_threshold = 0.35;

    std::ifstream file1(yolo_path.c_str());
    if (!file1.good()) {
        system("python3 src/utils/onnx2trt.py "
               "--onnx=model/ONNX/RM2025.onnx "
               "--saveEngine=model/TensorRT/yolo.engine "
               "--minBatch 1 "
               "--optBatch 1 "
               "--maxBatch 2 "
               "--Shape=1280x1280 "
               "--input_name=images");
    } else {
        TDT_INFO("Load yolo engine!");
    }
    std::ifstream file2(armor_path.c_str());
    if (!file2.good()) {
        system("python3 src/utils/onnx2trt.py "
               "--onnx=model/ONNX/armor_yolo.onnx "
               "--saveEngine=model/TensorRT/armor_yolo.engine "
               "--minBatch 1 "
               "--optBatch 5 "
               "--maxBatch 12 "
               "--Shape=192x192 "
               "--input_name=images");
    } else {
        TDT_INFO("Load armor_yolo engine!");
    }
    std::ifstream file3(classify_path.c_str());
    if (!file3.good()) {
        system("python3 src/utils/onnx2trt.py "
               "--onnx=model/ONNX/classify.onnx "
               "--saveEngine=model/TensorRT/classify.engine "
               "--minBatch 1 "
               "--optBatch 10 "
               "--maxBatch 20 "
               "--Shape=224x224 "
               "--input_name=input");
    } else {
        TDT_INFO("Load classify engine!");
    }
    std::cout << "yolo_path:" << yolo_path << "\n";
    std::cout << "armor_path:" << armor_path << "\n";
    std::cout << "classify_path:" << classify_path << "\n";
    this->classifier =
        classify::load(classify_path, classify::Type::densenet121);
    TDT_INFO("Load classify engine success!");

    this->armor_yolo =
        yolo::load(armor_path, yolo::Type::V8, 0.4f, 0.45f);
    TDT_INFO("Load armor_yolo engine success!");
    this->yolo = yolo::load(yolo_path, yolo::Type::V8, 0.65f, 0.45f);
    TDT_INFO("Load yolo engine success!");

    if (uav_mode) {
        std::ifstream file_uav(uav_yolo_path.c_str());
        if (!file_uav.good()) {
            RCLCPP_ERROR(this->get_logger(),
                         "UAV mode enabled but engine not found: %s",
                         uav_yolo_path.c_str());
            rclcpp::shutdown();
            return;
        }

        this->uav_yolo = yolo::load(
            uav_yolo_path, yolo::Type::V8,
            static_cast<float>(uav_confidence_threshold), 0.45f);
        TDT_INFO("Load UAV yolo engine success!");

        if (uav_forward_point2d) {
            point2d_pub = this->create_publisher<geometry_msgs::msg::Vector3>(
                "camera_point2D", rclcpp::SensorDataQoS());
            TDT_INFO("Enable UAV point2D forwarding.");
        } else {
            TDT_INFO("UAV point2D forwarding is disabled.");
        }
    }

    image_sub = this->create_subscription<sensor_msgs::msg::Image>(
        "camera_image", rclcpp::SensorDataQoS(),
        std::bind(&Detect::callback, this, std::placeholders::_1));
    pub = this->create_publisher<vision_interface::msg::DetectResult>(
        "detect_result", rclcpp::SensorDataQoS());
    debug_image_pub = this->create_publisher<sensor_msgs::msg::Image>(
        "detect_debug_image", rclcpp::SensorDataQoS());
    RCLCPP_INFO(this->get_logger(), "Detect node has been started.");
}

void Detect::callback(const std::shared_ptr<sensor_msgs::msg::Image> msg)
{
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();
    std::cout << "Detecting..." << std::endl;
    auto             img = cv_bridge::toCvShare(msg, "bgr8")->image;
    tdt_radar::Image image(img.data, img.cols, img.rows);

    if (uav_mode && uav_yolo) {
        cv::Rect uav_roi(0, 0, img.cols, img.rows);
        if (uav_use_right_half_roi && img.cols > 1) {
            uav_roi.x = img.cols / 2;
            uav_roi.width = img.cols - uav_roi.x;
        }

        cv::Mat uav_img = img(uav_roi).clone();
        tdt_radar::Image uav_image(uav_img.data, uav_img.cols, uav_img.rows);
        auto uav_result = uav_yolo->forward(uav_image);
        std::vector<yolo::Box> candidates;
        candidates.reserve(uav_result.size());
        for (auto& box : uav_result) {
            yolo::Box mapped_box = box;
            mapped_box.left += uav_roi.x;
            mapped_box.right += uav_roi.x;
            mapped_box.top += uav_roi.y;
            mapped_box.bottom += uav_roi.y;

            if (uav_target_class >= 0 && box.class_label != uav_target_class) {
                continue;
            }
            candidates.push_back(mapped_box);
        }

        if (!candidates.empty()) {
            auto best_it = std::max_element(
                candidates.begin(), candidates.end(),
                [](const yolo::Box& a, const yolo::Box& b) {
                    return a.confidence < b.confidence;
                });
            const auto& best_box = *best_it;

            if (uav_forward_point2d && point2d_pub) {
                geometry_msgs::msg::Vector3 point2d;
                point2d.x = (best_box.left + best_box.right) * 0.5f;
                point2d.y = (best_box.top + best_box.bottom) * 0.5f;
                point2d.z = best_box.confidence;
                point2d_pub->publish(point2d);
            }

            if (debug) {
                if (uav_use_right_half_roi) {
                    cv::line(img, cv::Point(uav_roi.x, 0),
                             cv::Point(uav_roi.x, img.rows - 1),
                             cv::Scalar(0, 200, 200), 2);
                }
                cv::rectangle(img,
                              cv::Rect(best_box.left, best_box.top,
                                       best_box.right - best_box.left,
                                       best_box.bottom - best_box.top),
                              cv::Scalar(0, 255, 255), 2);
                cv::putText(
                    img,
                    "uav cls=" + std::to_string(best_box.class_label) +
                        " conf=" + std::to_string(best_box.confidence),
                    cv::Point(best_box.left, std::max(0.0f, best_box.top - 10)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255),
                    2);
                cv::circle(img,
                           cv::Point((best_box.left + best_box.right) * 0.5f,
                                     (best_box.top + best_box.bottom) * 0.5f),
                           5, cv::Scalar(0, 255, 255), -1);
            }
        }
    }

    auto result = yolo->forward(image);
    if (result.size() == 0) {
        RCLCPP_INFO(this->get_logger(), "No Car!");
        return;
    } else if (result.size() > MAX_CARS) {
        RCLCPP_INFO(this->get_logger(), "Too Many Car!");
        return;
    }

    std::vector<tdt_radar::Image> images;
    std::vector<cv::Mat>          car_imgs;
    std::vector<Car>              cars;
    for (auto& box : result) {
        if (box.class_label == 0 || box.class_label == 1) {
            Car car;
            car.car = box;
            cars.push_back(car);
        }
    }

    for (auto& car : cars) {
        auto     temp_rect = cv::Rect(car.car.left, car.car.top,
                                      car.car.right - car.car.left,
                                      car.car.bottom - car.car.top);
        cv::Rect temp_car_rect = getSafeRect(img, temp_rect);
        auto     car_img = img(temp_car_rect);
        car_imgs.push_back(car_img.clone());
        car.car_rect = temp_car_rect;
    }

    for (auto& car_img : car_imgs) {
        auto image =
            tdt_radar::Image(car_img.data, car_img.cols, car_img.rows);
        images.push_back(image);
    }

    auto armor_boxes = armor_yolo->forwards(images);
    bool has_armor = false;
    for (int i = 0; i < armor_boxes.size(); i++) {
        if (armor_boxes[i].size() == 0) {
            continue;
        } else {
            cars[i].armors = armor_boxes[i];
            has_armor = true;
        }
    }  // 将armor_boxes存储到cars中
    if (!has_armor) {
        RCLCPP_INFO(this->get_logger(), "No Armor!");
        return;
    }
    // for(int i=0;i<cars.size();i++){
    //   if(cars[i].armors.size()==0){continue;}
    //   for(auto &armor:cars[i].armors){
    //     cv::Rect rect_img(
    //       armor.left+cars[i].car.left,
    //       armor.top+cars[i].car.top,
    //       armor.right-armor.left,
    //       armor.bottom-armor.top);
    //     // cv::rectangle(img,rect_img,cv::Scalar(255,255,255),2);
    //     cv::imwrite("/home/tdt/Label/armor_detect/"+std::to_string(count_img++)+".jpg",img(rect_img));
    //     }
    // }//这段是用来打印或者保存armor的图片
    std::vector<cv::Mat>         armor_imgs;
    std::vector<classify::Image> armor_images;

    for (auto& car : cars) {
        if (car.armors.size() == 0) {
            continue;
        }
        for (auto& armor : car.armors) {
            cv::Rect rect_img_1(
                armor.left + car.car.left, armor.top + car.car.top,
                armor.right - armor.left, armor.bottom - armor.top);
            cv::Rect rect_img = getSafeRect(img, rect_img_1);
            auto     armor_img = img(rect_img);
            armor_imgs.push_back(armor_img.clone());
        }
    }
    for (auto& armor_img : armor_imgs) {
        auto image =
            classify::Image(armor_img.data, armor_img.cols, armor_img.rows);
        armor_images.push_back(image);
    }
    auto armor_result = classifier->forwards(armor_images);

    // 保存用来训练分类
    //  for(int i=0;i<armor_imgs.size();i++){
    //    cv::imwrite("/home/mozihe/Label/fenqu_armor/"+std::to_string(armor_result[i])+"/"+std::to_string(count_img++)+".jpg",armor_imgs[i]);
    //  }

    for (auto& car : cars) {
        if (car.armors.size() == 0) {
            continue;
        }
        for (auto& armor : car.armors) {
            armor.class_label = armor_result[0];
            armor_result.erase(armor_result.begin());
            if (debug) {
                cv::putText(img, std::to_string(armor.class_label),
                            cv::Point(armor.left + car.car.left,
                                      armor.top + car.car.top),
                            cv::FONT_HERSHEY_SIMPLEX, 1,
                            cv::Scalar(255, 255, 255), 2);
            }
        }
    }
    vision_interface::msg::DetectResult detect_result;
    for (auto& car : cars) {
        if (car.armors.size() == 0) {
            continue;
        }
        cv::Rect max_rect;
        float    max_confidence = 0;
        for (auto& armor : car.armors) {
            if (armor.class_label != 0 && armor.class_label != 5 &&
                armor.confidence > max_confidence) {
                max_rect = cv::Rect(
                    armor.left + car.car.left, armor.top + car.car.top,
                    armor.right - armor.left, armor.bottom - armor.top);
                max_confidence = armor.confidence;
                car.number = armor.class_label;
            }
        }
        if (max_confidence == 0) {
            if (debug)
                cv::rectangle(img, car.car_rect, cv::Scalar(255, 255, 255),
                              2);
            continue;
        }
        auto safe_rect = getSafeRect(img, max_rect);
        auto max_mat = img(safe_rect);

        car.color = getColor(max_mat);
        car.center = cv::Point2f(max_rect.x + max_rect.width / 2,
                                 max_rect.y + max_rect.height / 2);
        if (car.color == 0) {
            detect_result.blue_x[car.number - 1] = car.center.x;
            detect_result.blue_y[car.number - 1] = car.center.y;
            if (car.center.x * car.center.y == 0 && car.number != 0) {
                RCLCPP_ERROR(this->get_logger(),
                             "Error: blue car center is 0");
            }
            if (debug) {
                cv::rectangle(img, car.car_rect, cv::Scalar(255, 0, 0), 2);
                cv::putText(img, std::to_string(car.car.confidence),
                            cv::Point(car.car.left, car.car.top),
                            cv::FONT_HERSHEY_SIMPLEX, 1,
                            cv::Scalar(255, 255, 255), 2);
            }
        }
        if (car.color == 2) {
            detect_result.red_x[car.number - 1] = car.center.x;
            detect_result.red_y[car.number - 1] = car.center.y;
            if (car.center.x * car.center.y == 0 && car.number != 0) {
                RCLCPP_ERROR(this->get_logger(),
                             "Error: red car center is 0");
            }
            if (debug) {
                cv::rectangle(img, car.car_rect, cv::Scalar(0, 0, 255), 2);
                cv::putText(img, std::to_string(car.car.confidence),
                            cv::Point(car.car.left, car.car.top),
                            cv::FONT_HERSHEY_SIMPLEX, 1,
                            cv::Scalar(255, 255, 255), 2);
            }
        }
        if (car.color == 1) {
            if (debug)
                cv::rectangle(img, car.car_rect, cv::Scalar(255, 255, 255),
                              2);
        }
    }
    detect_result.header.stamp = msg->header.stamp;
    pub->publish(detect_result);
    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used =
        std::chrono::duration_cast<std::chrono::duration<double>>(end -
                                                                  begin);
    std::cout << "Detect Time: " << time_used.count() * 1000 << "ms"
              << std::endl;
    cv::Mat final_img;
    cv::resize(img, final_img, cv::Size(1536, 1125));
    auto debug_msg = cv_bridge::CvImage(msg->header, "bgr8", final_img).toImageMsg();
    debug_image_pub->publish(*debug_msg);
    cv::imshow("detect", final_img);
    auto key = cv::waitKey(1);
    if (key == 'r') {
        debug = !debug;
    }
}
}  // namespace tdt_radar
RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::Detect)