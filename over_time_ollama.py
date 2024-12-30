from flask import Flask, request, jsonify 
from openai import OpenAI
import json
from datetime import datetime, timedelta
import pytz
import openai
import re
import ollama

app = Flask(__name__)

REGION_MAPPING = {
    "深圳": 1,
    "无锡": 2,
    "惠州": 3,
    "泰州": 4,
    "项目基地": 5,
    "其他": 6,
    "广州": 7,
    "西安": 8
}

class OvertimeParser:
    def __init__(self, model="qwen2:7b"):
        self.model = model
        # 设置时区为中国时区
        self.timezone = pytz.timezone('Asia/Shanghai')

    def get_current_time(self):
        """获取当前中国时间"""
        return datetime.now(self.timezone) 

    def round_up_to_half_hour(self, minutes):
        """将分钟数四舍五入到0.5小时为单位，确保加班时间至少1小时"""
        rounded_time = (minutes // 30) * 0.5 if minutes % 30 == 0 else ((minutes // 30) + 1) * 0.5
        return max(rounded_time, 1)

    def is_working_hours(self, check_time):
        """判断给定时间是否为正常工作时间"""
        # 工作时间：周一到周五的8:30-11:00 和 13:30-18:00
        if check_time.weekday() >= 5:  # 周六（5）和周日（6）
            return False
        
        # 上午8:30到11:00
        morning_start = check_time.replace(hour=8, minute=30, second=0, microsecond=0)  
        morning_end = check_time.replace(hour=11, minute=0, second=0, microsecond=0)
        
        # 下午1:30到6:00
        afternoon_start = check_time.replace(hour=13, minute=30, second=0, microsecond=0)
        afternoon_end = check_time.replace(hour=18, minute=0, second=0, microsecond=0)
        
        if morning_start <= check_time <= morning_end or afternoon_start <= check_time <= afternoon_end:
            return True
        
        return False

    def adjust_for_breaks(self, start_time, end_time):
        """调整加班时间，减去休息时间段"""
        breaks = [
            (start_time.replace(hour=12, minute=0), start_time.replace(hour=13, minute=30)),  # 中午休息
            (start_time.replace(hour=18, minute=0), start_time.replace(hour=19, minute=0))   # 晚间休息
        ]
        
        total_minutes = (end_time - start_time).seconds // 60  # 总分钟数
        for break_start, break_end in breaks:
            # 确保休息时间在加班时间范围内
            if break_start < end_time and break_end > start_time:
                # 计算重叠时间并从总分钟中减去
                overlap_start = max(start_time, break_start)
                overlap_end = min(end_time, break_end)
                overlap_minutes = (overlap_end - overlap_start).seconds // 60
                total_minutes -= overlap_minutes
        
        return total_minutes
    
    def align_time_to_five_minutes(self, time_str):
        # """将时间对齐到最近的五分钟倍数，向下取整"""
        time_obj = datetime.strptime(time_str, '%H:%M')
        minutes = time_obj.minute
        aligned_minutes = (minutes // 5) * 5
        if aligned_minutes != minutes:
            time_obj = time_obj.replace(minute=aligned_minutes)
            if time_obj.minute < minutes:
                pass  # 保持向下对齐
            else:
                pass  # 不需要向上调整
        return time_obj.strftime('%H:%M')

    def parse_overtime_info(self, text: str) -> dict:
        """
        解析加班信息文本，提取时间、地点、加班时长和事由。
        循环交互以补充缺失信息，直到所有字段完整。
        
        Args:
            text: 包含加班信息的文本。
            
        Returns:
            dict: 完整的加班信息，外层带True标记。
        """
        current_time = self.get_current_time()

        def generate_prompt(text, current_time):
            """生成解析Prompt"""
            return f"""
            请从以下文本中提取加班信息，并按照指定的JSON格式返回：
            文本：{text}
            
            当前时间信息：
            - 当前日期：{current_time.strftime('%Y-%m-%d')}
            - 当前时间：{current_time.strftime('%H:%M')}
            
            要求：
            1. 基于当前时间处理相对时间：
               - "明天"表示 {(current_time + timedelta(days=1)).strftime('%Y-%m-%d')}
               - "后天"表示 {(current_time + timedelta(days=2)).strftime('%Y-%m-%d')}
               - "今天"表示 {current_time.strftime('%Y-%m-%d')}
            2. 时间需转换为24小时制
            3. 如果没有提到事由，reason设为null
            4. 严格按照以下JSON格式返回，不要包含任何其他文字：
            {{
                "start_time": "HH:MM",
                "end_time": " HH:MM",
                "location": "地点",
                "overtimeDate": "加班当天的年月日%Y-%m-%d",
                "reason": "事由或null"
            }}

            其中
            """
        


        def validate_and_fill(result):
            """校验并返回缺失字段"""
            missing_fields = []
            if not result.get("start_time"):
                missing_fields.append("加班开始时间")
            if not result.get("end_time"):
                missing_fields.append("加班结束时间")
            if not result.get("location"):
                missing_fields.append("加班地点")
            if not result.get("sum_time"):
                result["sum_time"] = "1"  # 加班时间默认至少1小时
            if not result.get("reason") or result.get("reason") == "null":
                missing_fields.append("加班事由")
            return missing_fields

        incomplete = True
        result = {}
        while incomplete:
            prompt = generate_prompt(text, current_time)
            try:
                # 使用Ollama API
                stream = ollama.chat(
                    model=self.model,
                    messages=[{'role': 'user', 'content': prompt}],
                    stream=True
                )
                # 获取流数据并提取返回的内容
                response_text = ""
                for chunk in stream:
                    response_text += chunk['message']['content']
                
                print("返回的解析结果：", response_text)  # 打印返回的内容以调试
                # 尝试解析JSON
                try:
                    partial_result = json.loads(response_text)
                    # 合并新结果到 result，防止重复缺失
                    result.update(partial_result)
                    # 校验字段并判断是否完整
                    missing_fields = validate_and_fill(result)
                    if missing_fields:
                        # 提示用户补充缺失字段
                        print(f"以下字段信息缺失，请补充：{', '.join(missing_fields)}")
                        supplement_text = input("请输入补充内容：")
                        text = f"{text} {supplement_text}"  # 将补充内容拼接到原文本
                    else:
                        # 处理加班时间
                        start_time_str = result.get("start_time")
                        end_time_str = result.get("end_time")
                        if start_time_str and end_time_str:
                            start_time = datetime.strptime(start_time_str, '%H:%M')
                            end_time = datetime.strptime(end_time_str, '%H:%M')

                            # 对齐时间到最近的五分钟倍数
                            start_time_str = self.align_time_to_five_minutes(start_time_str)
                            end_time_str = self.align_time_to_five_minutes(end_time_str)

                            result["start_time"] = start_time_str
                            result["end_time"] = end_time_str

                            # 判断是否在正常工作时间内
                            if self.is_working_hours(start_time):
                                print("该时间段在正常工作时间内，请确认是否为加班。")
                                user_confirmation = input("请输入确认信息（是/否）：")
                                if user_confirmation != "是":
                                    return {"error": "该时间段为正常工作时间，未视为加班。"}

                            # 计算实际加班时长，扣除休息时间
                            adjusted_minutes = self.adjust_for_breaks(start_time, end_time)
                            sum_time = self.round_up_to_half_hour(adjusted_minutes)
                            result["sum_time"] = str(sum_time)

                        incomplete = False  # 所有字段已完整
                except json.JSONDecodeError as e:
                    print(f"JSON解析失败，请重新输入文本，错误信息: {str(e)}")
                    print(f"返回的文本内容: {response_text}")  # 打印返回的文本内容
                    text = input("请输入补充内容：")
            except Exception as e:
                print(f"API调用失败: {str(e)}")
                return {"error": f"API调用失败: {str(e)}"}

        # 返回完整结果，外层添加True标记
        result_with_flag = {
            "completed": True,
            "data": result
        }
        return result_with_flag

##############################


def transform_result(parsed_data):
    """
    将解析后的结果转换为新的输出格式。
    Args:
        parsed_data (dict): 原始解析结果
    Returns:
        dict: 转换后的结果
    """
    # print(f"Parsed Data: {parsed_data}") 
    location = parsed_data.get("location", "其他")  # 默认值为"其他"
    mapped_location = REGION_MAPPING.get(location.split()[0], 6)  # 根据地点映射获取编号
    overtime_date = parsed_data.get("overtimeDate")
    # start_time_str = parsed_data.get("start_time")
    # if start_time_str:
    #     start_time = datetime.strptime(start_time_str, '%H:%M')
    #     overtime_date = start_time.strftime('%Y-%m-%d')  # 获取加班当天的日期
    # else:
    #     overtime_date = datetime.now().strftime('%Y-%m-%d')  # 如果没有明确的开始时间，默认为当前日期

    # # start_time_str = parsed_data.get("start_time")
    # # overtime_date = datetime.now().strftime('%Y-%m-%d')  # 默认今天

    # 如果文本中提供了加班日期
    # date_match = re.search(r"(\d{4}-\d{2}-\d{2})", parsed_data.get("text", ""))
    # if date_match:
    #     overtime_date = date_match.group(1)  # 提取文本中的日期
    
    return {
        "completed": True,
        "data": {
            "cBgcc": mapped_location,
            "reason": parsed_data.get("reason"),
            "overtimeDate": overtime_date,  # 这里假定加班日期为今天
            "beginTime": parsed_data.get("start_time"),
            "endTime": parsed_data.get("end_time")
        }
    }
##############################################
@app.route('/')
def home():
    return "Hello, World!"

# API 路由
@app.route('/parse_overtime', methods=['POST'])
def parse_overtime():
    try:
        data = request.get_json()
        text = data.get("text")
        if not text:
            return jsonify({"error": "请提供加班信息文本"}), 400
        
        parser = OvertimeParser()
        # result = parser.parse_overtime_info(text)
        parsed_result = parser.parse_overtime_info(text)
        # if "error" in result:
        #     return jsonify(result), 400  # 提示用户错误
        if "error" in parsed_result:
            return jsonify(parsed_result), 400
        
        # return jsonify(result), 200  # 返回正常解析结果
                # 转换为新格式
        transformed_result = transform_result(parsed_result["data"])
        return jsonify(transformed_result), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
        parser = OvertimeParser()
        result = parser.parse_overtime_info(text)


        # return jsonify(result)
                # 将状态码和结果封装在一个字典中
        response = {
            # "completed": True,
            "data": result
        }

        # 使用 json.dumps() 来序列化 JSON 并设置 ensure_ascii=False
        response_json = json.dumps(response, ensure_ascii=False)

        # 返回 JSON 响应并指定状态码为 200
        return response_json, 200, {'Content-Type': 'application/json; charset=utf-8'}
        # # return jsonify(result, ensure_ascii=False), 200

        #         response_json = json.dumps(response, ensure_ascii=False)

        # # 返回 JSON 响应并指定状态码为 200
        # return response_json, 200, {'Content-Type': 'application/json'}
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        # print(response_json)
# 启动服务
if __name__ == "__main__":
    app.run(debug=True, host='192.168.3.156', port=6000)
