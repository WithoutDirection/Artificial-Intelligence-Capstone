# 指定編譯器
CXX = g++

# 指定編譯器選項
CXXFLAGS = -Wall -g

# 指定目標檔案名稱
TARGET = minmax_ver_3.exe

# 指定來源檔案
SOURCES = Sample.cpp STcpClient.h

# 指定目標規則
all: $(TARGET)

# 生成目標檔案的規則
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) -lws2_32

# 清除目標檔案和中間檔案的規則
clean:
	rm -f $(TARGET)