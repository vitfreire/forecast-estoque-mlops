from prefect import flow
from flows.training_flow import main as training_main
from flows.batch_inference_flow import main as batch_main


@flow(name="master_flow_walmart")
def main():
    train_out = training_main()
    batch_out = batch_main()

    return {
        "training": train_out,
        "batch": batch_out
    }


if __name__ == "__main__":
    main()