import socket
import struct

def get_robot_version():
# Tạo frame đúng theo định dạng
    transaction_id = 0x0001
    protocol_id = 0x0002
    length = 0x0001
    register = 0x01

    # Đóng gói dữ liệu theo dạng big-endian
    request = struct.pack('>HHHB', transaction_id, protocol_id, length, register)
    return request

# Tạo socket TCP
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("192.168.1.165", 502))  # Thay IP và cổng phù hợp

# Gửi request
sock.send(get_robot_version())

# Nhận phản hồi (giả sử tối đa 256 bytes)
response = sock.recv(256)
print("Response:", response.hex())

# Đóng kết nối
sock.close()

data = bytes.fromhex(response.hex())

# Phân tích phần header
transaction_id = int.from_bytes(data[0:2], "big")
protocol_id = int.from_bytes(data[2:4], "big")
length = int.from_bytes(data[4:6], "big")
register = data[6]
payload = data[7:7+length]

# In kết quả
print(f"Transaction ID: {transaction_id}")
print(f"Protocol ID: {protocol_id}")
print(f"Length: {length} bytes")
print(f"Register: {register}")
print(f"Payload (ASCII): {payload.rstrip(b'\x00').decode()}")
