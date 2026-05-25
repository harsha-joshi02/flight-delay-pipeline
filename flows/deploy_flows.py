"""Register Prefect deployments with scheduled runs."""

from prefect import serve
from prefect.schedules import CronSchedule

from flows.ingestion_flow import ingestion_flow
from flows.training_flow import training_flow
from flows.monitoring_flow import monitoring_flow


def main():
    ingestion_deployment = ingestion_flow.to_deployment(
        name="ingestion-scheduled",
        cron="0 2 1 * *",
        description="Monthly BTS data ingestion — 1st of month at 02:00 UTC",
    )

    training_deployment = training_flow.to_deployment(
        name="training-on-demand",
        description="Model training + promotion gate — triggered manually or by ingestion",
    )

    monitoring_deployment = monitoring_flow.to_deployment(
        name="monitoring-weekly",
        cron="0 6 * * 1",
        description="Weekly drift monitoring — every Monday at 06:00 UTC",
    )

    serve(ingestion_deployment, training_deployment, monitoring_deployment)


if __name__ == "__main__":
    main()
