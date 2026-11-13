FROM anpr-gpu:latest
COPY vehicle_tracker.py /app/vehicle_tracker.py
ENV TRACKER_BIND=0.0.0.0 TRACKER_PORT=8176 TRACKER_DB=/data/vehicle_tracker.db
EXPOSE 8176
VOLUME ["/data"]
CMD ["python", "vehicle_tracker.py"]
