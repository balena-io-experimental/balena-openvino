FROM openvino/model_server:2022.1
#CMD /ovms/bin/ovms --model_name ${MODEL_NAME} --model_path ${MODEL_PATH} --layout ${LAYOUT} --port 9000
ENTRYPOINT ["/ovms/bin/ovms --config_path /usr/models/config.json --port 9000" ]
#CMD bash
#[ "/ovms/bin/ovms", "--config_path", "/usr/models/config.json", "--port", "9000"]