import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
import argparse
import json

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
DEFAULT_CAMERA_INDEX = 1
DEFAULT_CAMERA_SERIAL_SUFFIX = "76630"


def list_realsense_devices():
    devices = []
    for index, device in enumerate(rs.context().query_devices()):
        devices.append(
            {
                "index": index,
                "serial": device.get_info(rs.camera_info.serial_number),
                "name": device.get_info(rs.camera_info.name),
                "product_line": device.get_info(rs.camera_info.product_line),
            }
        )
    return devices


def print_realsense_devices(devices, selected_serial=None):
    if not devices:
        print("没有检测到 RealSense 设备。")
        return
    print("检测到 RealSense 设备:")
    for device in devices:
        marker = "*" if selected_serial and device["serial"] == selected_serial else " "
        print(f" {marker} index={device['index']} serial={device['serial']} name={device['name']} product_line={device['product_line']}")


def select_realsense_device(devices, camera_serial=None, camera_index=DEFAULT_CAMERA_INDEX):
    if not devices:
        raise RuntimeError("没有找到 RealSense 相机。")
    if camera_serial:
        suffix_matches = []
        for device in devices:
            if device["serial"] == camera_serial:
                return device
            if device["serial"].endswith(camera_serial):
                suffix_matches.append(device)
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        if len(suffix_matches) > 1:
            available = ", ".join(f"{d['index']}:{d['serial']}" for d in suffix_matches)
            raise RuntimeError(f"RealSense 序列号后缀 {camera_serial} 匹配到多台设备: {available}")
        available = ", ".join(f"{d['index']}:{d['serial']}" for d in devices)
        raise RuntimeError(f"指定的 RealSense 序列号不存在: {camera_serial}. 可用设备: {available}")
    if camera_index < 0 or camera_index >= len(devices):
        available = ", ".join(f"{d['index']}:{d['serial']}" for d in devices)
        raise RuntimeError(f"指定的 RealSense index 越界: {camera_index}. 可用设备: {available}")
    return devices[camera_index]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Capture aligned RealSense RGB-D frames.")
    parser.add_argument("--camera-index", type=int, default=DEFAULT_CAMERA_INDEX, help="没有指定序列号时的备用 RealSense 设备 index。")
    parser.add_argument("--camera-serial", default=DEFAULT_CAMERA_SERIAL_SUFFIX, help="RealSense 完整序列号或唯一后缀；默认匹配 76630 结尾的相机。")
    parser.add_argument("--save-path", default="./data", help="保存 color/depth 的目录。")
    parser.add_argument("--list-devices", action="store_true", help="只列出 RealSense 设备，不启动相机。")
    return parser

def main():
    args = build_arg_parser().parse_args()

    # 0. 创建保存路径
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    # 1. 配置管线（Pipeline）
    devices = list_realsense_devices()
    if args.list_devices:
        print_realsense_devices(devices)
        return
    selected_device = select_realsense_device(devices, args.camera_serial, args.camera_index)
    print_realsense_devices(devices, selected_device["serial"])
    print(f"将使用相机: index={selected_device['index']} serial={selected_device['serial']}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(selected_device["serial"])

    # 配置彩色流和深度流：1280 * 720
    config.enable_stream(rs.stream.depth, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.bgr8, 30)

    # 2. 启动管线
    print("正在尝试启动 RealSense 相机...")
    try:
        profile = pipeline.start(config)
        print("相机启动成功！")
    except Exception as e:
        print(f"相机启动失败: {e}")
        print("请检查该相机是否支持 1280x720 depth/color，或换用 --camera-index / --camera-serial。")
        return

    # 创建对齐对象（将深度图对齐到彩色图，保证像素点一一对应）
    align_to = rs.stream.color
    align = rs.align(align_to)

    print("\n=========================================")
    print(" 操作说明:")
    print("  - 按 [ c ] 键：拍摄当前帧并保存到 ./data")
    print("  - 按 [ q ] 键：退出程序")
    print("=========================================\n")

    try:
        while True:
            # 3. 等待一帧数据并进行对齐
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            # 获取对齐后的深度帧和彩色帧
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # 4. 转换数据为 NumPy 数组
            # depth_image 是 uint16 类型（原始毫米级深度值）
            depth_image = np.asanyarray(depth_frame.get_data())
            # color_image 是 uint8 BGR 类型
            color_image = np.asanyarray(color_frame.get_data())

            # 5. 图像后处理（渲染深度图用于显示）
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )

            # 6. 图像缩放与拼接显示
            # 1280*720 横向拼接后为 2560*720。如果你的显示器放不下，可以取消下面三行的注释来缩小显示：
            # color_show = cv2.resize(color_image, (640, 360))
            # depth_show = cv2.resize(depth_colormap, (640, 360))
            # preview_images = np.hstack((color_show, depth_show))
            
            preview_images = np.hstack((color_image, depth_colormap))

            # 7. 显示窗口
            cv2.namedWindow('RealSense Control Panel', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense Control Panel', preview_images)

            # 8. 键盘事件监听
            key = cv2.waitKey(1) & 0xFF

            # 按 'q' 键退出
            if key == ord('q'):
                print("收到退出指令...")
                break

            # 按 'c' 键拍照
            elif key == ord('c'):
                # 使用时间戳命名，防止文件名冲突
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                
                color_filename = os.path.join(save_path, f"color_{timestamp}.png")
                depth_filename = os.path.join(save_path, f"depth_{timestamp}.png")
                meta_filename = os.path.join(save_path, f"camera_meta_{timestamp}.json")
                
                # 保存彩色图 (BGR)
                cv2.imwrite(color_filename, color_image)
                
                # 保存原始深度图 (OpenCV 支持直接将 uint16 单通道数组保存为 16位 PNG)
                # 这样可以完整保留相机的毫米级深度数据，而不是损失精度的彩色伪彩图
                cv2.imwrite(depth_filename, depth_image)

                color_intr = color_frame.profile.as_video_stream_profile().intrinsics
                depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
                meta = {
                    "selected_device": selected_device,
                    "available_devices": devices,
                    "width": IMAGE_WIDTH,
                    "height": IMAGE_HEIGHT,
                    "depth_scale_m": float(depth_scale),
                    "color_path": os.path.abspath(color_filename),
                    "depth_path": os.path.abspath(depth_filename),
                    "intrinsics": {
                        "fx": float(color_intr.fx),
                        "fy": float(color_intr.fy),
                        "cx": float(color_intr.ppx),
                        "cy": float(color_intr.ppy),
                        "model": str(color_intr.model),
                        "coeffs": [float(value) for value in color_intr.coeffs],
                    },
                }
                with open(meta_filename, "w", encoding="utf-8") as file:
                    json.dump(meta, file, ensure_ascii=False, indent=2)
                
                print(f"拍照成功！已保存到:")
                print(f"   -> {color_filename}")
                print(f"   -> {depth_filename}")
                print(f"   -> {meta_filename}")

    except Exception as e:
        print(f"运行过程中发生异常: {e}")

    finally:
        # 9. 释放资源
        pipeline.stop()
        cv2.destroyAllWindows()
        print("管线已安全关闭。")

if __name__ == "__main__":
    main()
