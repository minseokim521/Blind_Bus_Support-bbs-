import psycopg2
import requests
import xml.etree.ElementTree as ET



class API():
    def __init__(self, Bus_num, Station_name):
        self.Bus_num = Bus_num
        self.Station_name = Station_name
            
    def database_query(self, Bus_num, Station_name):
        #데이터베이스에 연결시도
        try:
            # 데이터베이스 연결 & 커넥트 객체 생성
            conn = psycopg2.connect(host="127.0.0.1", dbname="postgres", user="postgres", password="postgres")
        except:
            print("Not Connected!.")

        # 쿼리를 수행하는 cursor객체 생성
        cursor = conn.cursor()

        # 쿼리문
        sql_bus = "SELECT routeid FROM bus WHERE bus_id = %s;"
        sql_station = "SELECT node_id FROM station WHERE station_name = %s;"
        # 쿼리실행
        cursor.execute(sql_bus, (Bus_num,))
        #fetchone은 쿼리에 해당하는 열을 튜플형태로 반환
        bus_result = cursor.fetchone()

        # 예외 처리
        if bus_result:
            print("Bus Route ID :", bus_result[0])
        else:
            print("No bus found with the given bus_id")


        cursor.execute(sql_station, (Station_name,))

        station_result = cursor.fetchone()

        if station_result:
            print("Staion ID :", station_result[0])
        else:
            print("No Station found with the given Station_name")

        # 데이터베이스 연결 끊기
        conn.close()
        return bus_result, station_result
    
    # 특정 정류장의 정보(그 정류장에서 운행하는 버스들, 도착정보)
    def station_bus_list(self, station_result):
        url = 'http://ws.bus.go.kr/api/rest/arrive/getLowArrInfoByStId'
        service_key = "lnGvRUsSrOgezp/xjHmRf1XJipLQd9ANFdkUk5w2kB1FaTDTAcS88zmKBViC6HYFRcWfhGjkuNQD85aNrvoTTw=="

        params ={'serviceKey' : service_key,
                'stId' : str(station_result) }

        response = requests.get(url, params=params)
        return response.content


    #  특정 버스 노선이 경유하는 버스 정류소의 정보
    def bus_station_list(self, bus_result):
        url = 'http://ws.bus.go.kr/api/rest/busRouteInfo/getStaionByRoute'
        service_key = "lnGvRUsSrOgezp/xjHmRf1XJipLQd9ANFdkUk5w2kB1FaTDTAcS88zmKBViC6HYFRcWfhGjkuNQD85aNrvoTTw=="

        params ={'serviceKey' : service_key,
                'busRouteId' : str(bus_result) }

        response = requests.get(url, params=params)
        return response.content


    # 정류소 노선별 교통약자 도착예정정보
    def station_arrival_info(self, station_result, bus_result, ord):
        url = 'http://ws.bus.go.kr/api/rest/arrive/getLowArrInfoByRoute'
        service_key = "lnGvRUsSrOgezp/xjHmRf1XJipLQd9ANFdkUk5w2kB1FaTDTAcS88zmKBViC6HYFRcWfhGjkuNQD85aNrvoTTw=="


        params ={'serviceKey' : service_key,
                'stId' : str(station_result),
                'busRouteId' : str(bus_result),
                'ord' : str(ord) }

        response = requests.get(url, params=params)
        return response.content

    # xml 값에서 특정 val라는 tag안에 있는 item을 가져오는 함수
    def find_xml_val(self, root, val):
        item_list = []
        for item in root.findall(".//itemList"):
            item1 = item.find(str(val)).text  # val에 해당하는 태그의 텍스트 내용을 가져옵니다.
            item_list.append(item1)
        return item_list

    # 리스트에서 일치하는 값을 찾아서 True값과 인덱스를 반환해준다.
    def find_api_val(self, list, value_to_find):
        for i, val in enumerate(list):
            if str(val)==str(value_to_find):
                print(f"value {value_to_find} found at index {i}")
                return i, True
        print("could not found the value in the list")
        return None, False
