#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include <cstring>

namespace py = pybind11;

/**
 * @brief Project 3D LiDAR points onto a 2D top-down occupancy grid.
 *
 * Converts a set of 3D points into a discretised bird’s-eye-view (BEV)
 * representation. Each valid point is mapped to a grid cell and marked
 * as occupied.
 *
 * @param points Input NumPy array of shape [N, 4] (x, y, z, intensity).
 * @param grid_size Resolution of the square output grid.
 * @param range_max Maximum sensing range in metres.
 *
 * @return 2D occupancy grid (uint8), where occupied cells are 255 and
 *         empty cells are 0.
 *
 * @note Assumes LiDAR points are expressed in the vehicle coordinate
 *       frame (x forward, y lateral, z upward).
 *
 * @performance O(N) over the number of input points.
 */
py::array_t<uint8_t> project_to_2d_points(py::array_t<float> points, int grid_size, float range_max) {
    auto buf = points.request();
    float* ptr = static_cast<float*>(buf.ptr);
    int num_points = buf.shape[0];

    // Initialise empty grid
    auto grid = py::array_t<uint8_t>({grid_size, grid_size});
    auto grid_buf = grid.request();
    uint8_t* grid_ptr = static_cast<uint8_t*>(grid_buf.ptr);
    std::memset(grid_ptr, 0, grid_size * grid_size * sizeof(uint8_t));

    // Metres per grid cell
    float resolution = (2.0f * range_max) / grid_size;

    for (int i = 0; i < num_points; ++i) {
        float x = ptr[i * 4 + 0]; ///< Forward axis (metres, vehicle frame)
        float y = ptr[i * 4 + 1]; ///< Lateral axis (metres, vehicle frame)

        int gi = static_cast<int>((x + range_max) / resolution);
        int gj = static_cast<int>((y + range_max) / resolution);

        if (gi >= 0 && gi < grid_size && gj >= 0 && gj < grid_size) {
            grid_ptr[gi * grid_size + gj] = 255;
        }
    }

    return grid;
}

/**
 * @brief Convert a raw CARLA camera buffer into an RGB image.
 *
 * Transforms an input BGRA image buffer into RGB format and resizes it
 * to a target resolution suitable for downstream tasks such as
 * reinforcement learning or perception pipelines.
 *
 * @param input Input NumPy array of shape [H, W, 4] (BGRA format).
 * @param target_w Desired output width in pixels.
 * @param target_h Desired output height in pixels.
 *
 * @return NumPy array (uint8) of shape [target_h, target_w, 3]
 *         containing the processed RGB image.
 *
 * @throws std::runtime_error If the input buffer is invalid or empty.
 */
py::array_t<uint8_t> process_camera_buffer(py::array_t<uint8_t> input, int target_w, int target_h) {
    auto buf = input.request();

    if (buf.size == 0 || buf.ndim < 2) {
        throw std::runtime_error("Camera input buffer is empty or invalid");
    }

    int h = buf.shape[0];
    int w = buf.shape[1];

    // Create OpenCV matrix (BGRA)
    cv::Mat img(h, w, CV_8UC4, (void*)buf.ptr);

    if (img.empty()) {
        throw std::runtime_error("Failed to construct OpenCV matrix from buffer");
    }

    // Convert to RGB
    cv::Mat rgb_img;
    cv::cvtColor(img, rgb_img, cv::COLOR_BGRA2RGB);

    // Resize image
    cv::Mat resized_img;
    cv::resize(rgb_img, resized_img,
               cv::Size(target_w, target_h),
               0, 0, cv::INTER_AREA);

    // Copy to NumPy output
    auto result = py::array_t<uint8_t>({target_h, target_w, 3});
    std::memcpy(
        result.request().ptr,
        resized_img.data,
        target_h * target_w * 3
    );

    return result;
}

/**
 * @brief Render a combined visualisation of camera and LiDAR data.
 *
 * Displays a side-by-side view consisting of:
 * - Left: RGB camera feed (converted from BGRA)
 * - Right: LiDAR bird’s-eye-view (BEV) projection
 *
 * LiDAR points are colour-mapped according to their Euclidean distance
 * from the sensor, producing a gradient (red for near, green for far).
 *
 * @param camera_input Raw camera buffer [H, W, 4] (BGRA).
 * @param lidar_points LiDAR point cloud [N, 4].
 * @param range_max Maximum sensing range in metres.
 *
 * @note Points outside the specified range are ignored.
 */
void draw_sensors(py::array_t<uint8_t> camera_input, py::array_t<float> lidar_points, float range_max) {

    auto cam_buf = camera_input.request();
    int cam_h = cam_buf.shape[0];
    int cam_w = cam_buf.shape[1];

    // Convert camera to BGR for OpenCV display
    cv::Mat raw_cam(cam_h, cam_w, CV_8UC4, (void*)cam_buf.ptr);
    cv::Mat left_view;
    cv::cvtColor(raw_cam, left_view, cv::COLOR_BGRA2BGR);

    int width = 640;
    int height = 480;

    cv::resize(left_view, left_view, cv::Size(width, height));

    // Initialise LiDAR canvas
    cv::Mat right_view = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);

    auto lidar_buf = lidar_points.request();
    float* ptr = static_cast<float*>(lidar_buf.ptr);

    float scale = static_cast<float>(height) / (2.0f * range_max);

    for (int i = 0; i < lidar_buf.shape[0]; ++i) {
        float x = ptr[i * 4 + 0]; ///< Forward axis (metres)
        float y = ptr[i * 4 + 1]; ///< Lateral axis (metres)

        float dist = std::sqrt(x * x + y * y);
        float normalised_dist = std::min(1.0f, dist / range_max);

        float hue = normalised_dist * 120.0f;

        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(static_cast<int>(hue), 255, 255));
        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

        cv::Scalar colour(bgr.data[0], bgr.data[1], bgr.data[2]);

        int px = width / 2 + static_cast<int>(y * scale);
        int py = height / 2 - static_cast<int>(x * scale);

        if (px >= 0 && px < width && py >= 0 && py < height) {
            cv::circle(right_view, cv::Point(px, py), 1, colour, -1);
        }
    }

    // Combine views
    cv::Mat canvas = cv::Mat::zeros(height, width * 2, CV_8UC3);

    left_view.copyTo(canvas(cv::Rect(0, 0, width, height)));
    right_view.copyTo(canvas(cv::Rect(width, 0, width, height)));

    // Labels
    cv::putText(canvas, "CAMERA",
        cv::Point(20, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.7,
        cv::Scalar(255, 255, 255), 2
    );

    cv::putText(canvas, "LIDAR (BEV)",
        cv::Point(width + 20, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.7,
        cv::Scalar(255, 255, 255), 2
    );

    cv::imshow("Agent Perception Hub", canvas);
    cv::waitKey(1);
}

/**
 * @brief Python module definition for sensor utilities.
 *
 * Exposes high-performance C++ implementations to Python via Pybind11,
 * including LiDAR projection, camera preprocessing, and visualisation.
 */
PYBIND11_MODULE(sensor_utils, m) {
    m.def("project_to_2d_points", &project_to_2d_points,
          "Project LiDAR points to a 2D occupancy grid");

    m.def("process_camera_buffer", &process_camera_buffer,
          "Convert BGRA image to RGB and resize");

    m.def("draw_sensors", &draw_sensors,
          "Render camera and LiDAR visualisation");
}