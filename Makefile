image_name=voice-dipatcher
container_name=voice-dipatcher-container

build:
	docker build -t $(image_name) .

run:
	docker run -d --name $(container_name) \
		--device /dev/snd \
		-p 5000:5000 \
		$(image_name)

stop:
	docker stop $(container_name) && docker rm $(container_name)

logs:
	docker logs -f $(container_name)

restart: stop run