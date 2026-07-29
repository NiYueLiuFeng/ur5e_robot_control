from spnav import spnav_open, spnav_poll_event, spnav_close
from spnav import SpnavMotionEvent, SpnavButtonEvent
import time

if spnav_open() == -1:
    print("错误：无法连接到 spacenavd。")
    print("请确保服务已启动 (sudo service spacenavd start)")
    exit(1)

print("成功连接到 3D 鼠标。开始读取数据...")
print("提示：按压摇杆或按下按钮，数据将在此显示。按 Ctrl+C 退出。\n")

try:
    while True:
        # 2. 轮询事件（非阻塞）
        event = spnav_poll_event()

        # 3. 处理运动事件
        if isinstance(event, SpnavMotionEvent):
            # event.translation: 一个包含 (x, y, z) 的三元组，表示平移力
            # event.rotation: 一个包含 (rx, ry, rz) 的三元组，表示旋转力矩

            axes_xyz = [v / 350.0 for v in event.translation]
            # 限制数值范围在 [-1.0, 1.0]
            axes_xyz = [max(-1.0, min(1.0, val)) for val in axes_xyz]

            axes_rt = [v / 350.0 for v in event.rotation]
            # 限制数值范围在 [-1.0, 1.0]
            axes_rt = [max(-1.0, min(1.0, val)) for val in axes_rt]
        
            print(f"运动 - 平移: {event.translation}, 旋转: {event.rotation}")

        # 4. 处理按钮事件
        elif isinstance(event, SpnavButtonEvent):
            state = "按下" if event.press else "释放"
            print(f"按钮 - 编号 {event.bnum}: {state}")

        # 短暂休眠，避免循环占用过多 CPU
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n用户中断。正在清理并关闭连接...")

finally:
    # 5. 关闭连接
    spnav_close()
    print("连接已关闭。")