import grpc
import Smoke2d_pb2
import Smoke2d_pb2_grpc
import numpy as np
import matplotlib.pyplot as plt


class Smoke2dClient(object):
    PROPERTY_DENSITY = 1
    PROPERTY_VELOCITY = 4

    def __init__(self, channel):
        self._stub = Smoke2d_pb2_grpc.Smoke2dStub(channel)

    def init(self, nx, ny):
        params = Smoke2d_pb2.Smoke2dInitParams(nx=nx, ny=ny)
        result = self._stub.Init(params)
        self.printStatus("init", result.status)

    def step(self):
        params = Smoke2d_pb2.Smoke2dStepParams()
        result = self._stub.Step(params)
        self.printStatus("step", result.status)

    def reset(self):
        params = Smoke2d_pb2.Smoke2dResetParams()
        result = self._stub.Reset(params)
        self.printStatus("reset", result.status)

    def destroy(self):
        params = Smoke2d_pb2.Smoke2dDestroyParams()
        result = self._stub.Destroy(params)
        self.printStatus("destroy", result.status)

    def get_data(self, property):
        params = Smoke2d_pb2.Smoke2dGetDataParams(property=property)
        data = b''
        for data_chunk in self._stub.GetData(params):
            data += data_chunk.data
        return data

    def get_data_to_view(self, property, view):
        params = Smoke2d_pb2.Smoke2dGetDataParams(property=property)
        for data_chunk in self._stub.GetData(params):
            size = len(data_chunk.data)
            view[:size] = data_chunk.data
            view = view[size:]

    def printStatus(self, name, status):
        # print("Function", name, "returned with status", status)
        pass


def main():
    channel = grpc.insecure_channel('localhost:50077')
    client = Smoke2dClient(channel=channel)
    client.init(64, 64)

    plt.ion()
    plt.gcf().set_size_inches(14, 7)

    frame = 0
    while True:
        plt.clf()
        rh = np.frombuffer(client.getData(), dtype=np.float32)
        rh = rh.reshape(64, 64)
        plt.subplot(1, 2, 1)
        plt.imshow(rh)
        plt.axis('equal')
        u = np.frombuffer(client.getData(), dtype=np.float32)
        u = u.reshape(64, 64, 2)
        plt.subplot(1, 2, 2)
        plt.quiver(u[:, :, 0], u[:, :, 1])
        plt.axis('equal')
        print(frame)

        frame += 1
        if frame % 100 == 0:
            client.reset()
        else:
            client.step()
        plt.pause(0.01)



if __name__ == "__main__":
    main()
