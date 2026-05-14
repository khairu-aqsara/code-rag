.PHONY: up down restart logs test test-unit test-integration clean

up:
	docker-compose up -d

down:
	docker-compose down

restart:
	docker-compose restart

logs:
	docker-compose logs -f

test:
	pytest -v

test-unit:
	pytest -m "not integration" -v

test-integration:
	pytest -m integration -v

clean:
	docker-compose down -v