
# response1 = bus_api.station_pose(X_location, Y_locatioin, radius)

# # #xml 값 가져옴
# root1 = ET.fromstring(response1)

# # #가장 가까운 버스정류장 선택
# station_list = bus_api.find_xml_val(root1, "arsId")
# print(station_list)

# ## 첫번째 정류소가 가장 가까울 것이라고 가정
# routeid = station_list[0]
