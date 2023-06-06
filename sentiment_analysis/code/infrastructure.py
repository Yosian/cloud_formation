import boto3


class HandleS3Objects:
    def __init__(self, bucket: str, origin: str, destination: str):
        self.bucket = bucket
        self.origin = origin
        self.destination = destination
        self.s3 = boto3.client('s3')

    def obtain_file_from_bucket(self) -> None:
        self.s3.download_file(Bucket=self.bucket,
                              Key=self.origin,
                              Filename=self.destination)

    def upload_file_to_bucket(self) -> None:
        self.s3.upload_file(Filename=self.origin,
                            Bucket=self.bucket,
                            Key=self.destination)