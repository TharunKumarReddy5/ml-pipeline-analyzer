from diagrams import Cluster, Diagram
from diagrams.gcp.analytics import BigQuery, Dataflow, PubSub
from diagrams.gcp.compute import AppEngine, Functions
from diagrams.gcp.database import BigTable
from diagrams.gcp.storage import GCS

with Diagram("Media Monitoring Storage Architecture", show=False) as med_diag:
    pubsub = PubSub("pubsub")
    flow = Dataflow("DataFlow")

    with Cluster("Data Collection"):
        [Functions("RSS Feed Webhook"),
         Functions("Twitter Webhook"),
         Functions("Press Release")] >> pubsub >> flow

    with Cluster("Storage"):
        with Cluster("Data Lake"):
            flow >> [BigQuery("BigQuery"),
                     GCS("Storage")]

        with Cluster("Event Driven"):
            with Cluster("Processing"):
                flow >> AppEngine("GAE") >> BigTable("BigTable")

            with Cluster("Serverless"):
                flow >> Functions("Function") >> AppEngine("AppEngine")

    pubsub >> flow
med_diag
