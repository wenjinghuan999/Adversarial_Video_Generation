import grpc
from client import Smoke2dClient
import numpy as np
import matplotlib.pyplot as plt
import threading
import random

DEBUG_OUTPUT = False


class DataFetcher(object):

    def __init__(self, host, port):
        channel = grpc.insecure_channel('{0}:{1}'.format(host, port))
        self.client = Smoke2dClient(channel=channel)
        self.client.init(64, 64)
        if DEBUG_OUTPUT:
            print('Connected:', '{0}:{1}'.format(host, port))
        self.frame = 0

    def fetch(self):
        if self.frame % 100 == 0:
            self.client.reset()
        else:
            self.client.step()
        rh = np.frombuffer(self.client.get_data(Smoke2dClient.PROPERTY_DENSITY), dtype=np.float32)
        rh = rh.reshape(64, 64)
        u = np.frombuffer(self.client.get_data(Smoke2dClient.PROPERTY_VELOCITY), dtype=np.float32)
        u = u.reshape(64, 64, 2)
        if DEBUG_OUTPUT:
            print('Data fetched: frame', self.frame)
        self.frame += 1
        return rh, u

    def fetch_one(self):
        self.client.reset()
        rh = np.frombuffer(self.client.get_data(Smoke2dClient.PROPERTY_DENSITY), dtype=np.float32)
        rh = rh.reshape(1, 64, 64, 1)
        u = np.frombuffer(self.client.get_data(Smoke2dClient.PROPERTY_VELOCITY), dtype=np.float32)
        u = u.reshape(1, 64, 64, 2)
        if DEBUG_OUTPUT:
            print('Data fetched')
        return rh, u

    def fetch_episode(self, episode_length=100):
        self.client.reset()
        rh = bytearray(episode_length * 64 * 64 * 4)
        u = bytearray(episode_length * 64 * 64 * 2 * 4)
        view_rh = memoryview(rh)
        view_u = memoryview(u)

        i = 0
        while i < episode_length:
            self.client.get_data_to_view(view_rh)
            view_rh = view_rh[64 * 64 * 4:]
            self.client.get_data_to_view(view_u)
            view_u = view_u[64 * 64 * 4 * 2:]
            i += 1
            self.client.step()

        rh = np.frombuffer(bytes(rh), dtype=np.float32)
        rh = rh.reshape([episode_length, 64, 64, 1])
        u = np.frombuffer(bytes(u), dtype=np.float32)
        u = u.reshape([episode_length, 64, 64, 2])
        if DEBUG_OUTPUT:
            print('Data batch fetched:', episode_length)
        return rh, u

    def fetch_batch(self, n=1000):
        rh1 = bytearray(n * 64 * 64 * 4)
        u1 = bytearray(n * 64 * 64 * 2 * 4)
        rh2 = bytearray(n * 64 * 64 * 4)
        u2 = bytearray(n * 64 * 64 * 2 * 4)
        view_rh = memoryview(rh1)
        view_u = memoryview(u1)
        view_rh2 = memoryview(rh2)
        view_u2 = memoryview(u2)

        i = 0
        while i < n:
            self.client.reset()

            while True:
                try:
                    self.client.get_data_to_view(Smoke2dClient.PROPERTY_DENSITY, view_rh)
                    break
                except:
                    continue
            view_rh = view_rh[64 * 64 * 4:]
            self.client.get_data_to_view(Smoke2dClient.PROPERTY_VELOCITY, view_u)
            view_u = view_u[64 * 64 * 4 * 2:]

            # self.client.step()
            #
            # self.client.get_data_to_view(view_rh2)
            # view_rh2 = view_rh2[64 * 64 * 4:]
            # self.client.get_data_to_view(view_u2)
            # view_u2 = view_u2[64 * 64 * 4 * 2:]
            i += 1

        rh1 = np.frombuffer(bytes(rh1), dtype=np.float32)
        rh1 = rh1.reshape([n, 64, 64, 1])
        u1 = np.frombuffer(bytes(u1), dtype=np.float32)
        u1 = u1.reshape([n, 64, 64, 2])
        # rh2 = np.frombuffer(bytes(rh2), dtype=np.float32)
        # rh2 = rh2.reshape([n, 64, 64, 1])
        # u2 = np.frombuffer(bytes(u2), dtype=np.float32)
        # u2 = u2.reshape([n, 64, 64, 2])
        if DEBUG_OUTPUT:
            print('Data batch fetched:', n)
        # return rh1, u1, rh2, u2
        return rh1, u1, rh1, u1

    def fetch_episodes(self, batch_size, episode_length, in_serial=True, skip=None):
        rh = bytearray(batch_size * episode_length * 64 * 64 * 4)
        u = bytearray(batch_size * episode_length * 64 * 64 * 2 * 4)
        view_rh = memoryview(rh)
        view_u = memoryview(u)

        for j in range(batch_size):
            self.client.reset()
            if in_serial and not skip is None:
                for i in range(skip[j]):
                    self.client.step()

            for i in range(episode_length):
                while True:
                    try:
                        self.client.get_data_to_view(Smoke2dClient.PROPERTY_DENSITY, view_rh)
                        break
                    except:
                        continue
                view_rh = view_rh[64 * 64 * 4:]
                self.client.get_data_to_view(Smoke2dClient.PROPERTY_VELOCITY, view_u)
                view_u = view_u[64 * 64 * 4 * 2:]
                if in_serial:
                    self.client.step()
                else:
                    self.client.reset()

        rh = np.frombuffer(bytes(rh), dtype=np.float32)
        rh = rh.reshape([batch_size, episode_length, 64, 64, 1])
        u = np.frombuffer(bytes(u), dtype=np.float32)
        u = u.reshape([batch_size, episode_length, 64, 64, 2])
        if DEBUG_OUTPUT:
            print('Data episodes fetched:', batch_size, 'x', episode_length)
            print('Skip:', skip)
        return rh, u

    def fetch_episodes_dynamic_length(self, batch_size, max_length):
        rh = bytearray(batch_size * max_length * 64 * 64 * 4)
        u = bytearray(batch_size * max_length * 64 * 64 * 2 * 4)
        view_rh = memoryview(rh)
        view_u = memoryview(u)

        lengths = np.array([[random.randint(5, max_length // 2), random.randint(5, max_length // 2)]
                            for _ in range(batch_size)])
        lengths = np.array([[5, max_length // 2] for _ in range(batch_size)])

        for j in range(batch_size):
            self.client.reset()
            for i in range(random.randint(0, max_length // 8)):
                self.client.step()

            for i in range(lengths[j, 0] + lengths[j, 1]):
                while True:
                    try:
                        self.client.get_data_to_view(Smoke2dClient.PROPERTY_DENSITY, view_rh)
                        break
                    except:
                        continue
                view_rh = view_rh[64 * 64 * 4:]
                self.client.get_data_to_view(Smoke2dClient.PROPERTY_VELOCITY, view_u)
                view_u = view_u[64 * 64 * 4 * 2:]
                self.client.step()

            for i in range(lengths[j, 0] + lengths[j, 1], max_length):
                view_rh = view_rh[64 * 64 * 4:]
                view_u = view_u[64 * 64 * 4 * 2:]

        rh = np.frombuffer(bytes(rh), dtype=np.float32)
        rh = rh.reshape([batch_size, max_length, 64, 64, 1])
        u = np.frombuffer(bytes(u), dtype=np.float32)
        u = u.reshape([batch_size, max_length, 64, 64, 2])
        if DEBUG_OUTPUT:
            print('Data episodes fetched:', batch_size, 'x', np.mean(lengths))

        return rh, u, lengths


class CachedDataFetcher:

    def __init__(self, host, port):
        self.fetcher = DataFetcher(host, port)
        self.next_batch = None
        self.next_batch_n = None
        self.t = None

    def worker(self, n=1000):
        rh1, u1, rh2, u2 = self.fetcher.fetch_batch(n)
        self.next_batch_n = n
        self.next_batch = [rh1, u1, rh2, u2]
        if DEBUG_OUTPUT:
            print('Next batch ready({0})'.format(n))
        self.t = None

    def worker_episodes(self, batch_size, episode_length, in_serial=True, skip=None):
        rh, u = self.fetcher.fetch_episodes(batch_size, episode_length, in_serial, skip)
        self.next_batch_n = batch_size, episode_length
        self.next_batch = rh, u
        if DEBUG_OUTPUT:
            print('Next batch ready({0}x{1})'.format(batch_size, episode_length))
        self.t = None

    def worker_episodes_dynamic_length(self, batch_size, max_length):
        next_batch = self.fetcher.fetch_episodes_dynamic_length(batch_size, max_length)
        self.next_batch_n = batch_size, max_length
        self.next_batch = next_batch
        if DEBUG_OUTPUT:
            print('Next batch ready({0}x{1})'.format(batch_size, max_length))
        self.t = None

    def fetch_one(self):
        if self.t:
            if DEBUG_OUTPUT:
                print('Waiting')
            try:
                self.t.join()
            except:
                pass
        return self.fetcher.fetch_one()

    def fetch_batch(self, n=1000):
        if self.next_batch and self.next_batch_n == n:
            this_batch = self.next_batch
        elif self.t:
            if DEBUG_OUTPUT:
                print('Waiting')
            try:
                self.t.join()
            except:
                pass
            this_batch = self.next_batch
        else:
            this_batch = self.fetcher.fetch_batch(n)

        self.next_batch = None
        self.t = threading.Thread(target=CachedDataFetcher.worker, args=(self, n))
        self.t.start()

        return this_batch

    def fetch_episodes(self, batch_size, episode_length, in_serial=True, skip=None):
        if self.next_batch and self.next_batch_n == batch_size:
            this_batch = self.next_batch
        elif self.t:
            if DEBUG_OUTPUT:
                print('Waiting')
            try:
                self.t.join()
            except:
                pass
            this_batch = self.next_batch
        else:
            this_batch = self.fetcher.fetch_episodes(batch_size, episode_length, in_serial, skip)

        self.next_batch = None
        self.t = threading.Thread(target=CachedDataFetcher.worker_episodes,
                                  args=(self, batch_size, episode_length, in_serial, skip))
        self.t.start()

        return this_batch

    def fetch_episodes_dynamic_length(self, batch_size, max_length):
        if self.next_batch and self.next_batch_n == batch_size:
            this_batch = self.next_batch
        elif self.t:
            if DEBUG_OUTPUT:
                print('Waiting')
            try:
                self.t.join()
            except:
                pass
            this_batch = self.next_batch
        else:
            this_batch = self.fetcher.fetch_episodes_dynamic_length(batch_size, max_length)

        self.next_batch = None
        self.t = threading.Thread(target=CachedDataFetcher.worker_episodes_dynamic_length,
                                  args=(self, batch_size, max_length))
        self.t.start()

        return this_batch


def main():
    fetcher = CachedDataFetcher('localhost', 50077)

    plt.ion()
    plt.gcf().set_size_inches(9, 9)

    n = 10
    while True:
        rh1, u1, rh2, u2 = fetcher.fetch_batch(n)
        frame = 0
        while frame < n:
            plt.clf()
            plt.subplot(2, 2, 1)
            plt.imshow(rh1[frame, ..., 0])
            plt.axis('equal')
            plt.subplot(2, 2, 2)
            plt.quiver(u1[frame, ..., 0], u1[frame, ..., 1])
            plt.axis('equal')
            plt.subplot(2, 2, 3)
            plt.imshow(rh2[frame, ..., 0])
            plt.axis('equal')
            plt.subplot(2, 2, 4)
            plt.quiver(u2[frame, ..., 0], u2[frame, ..., 1])
            plt.axis('equal')
            print(frame)
            print(np.max(rh1), np.min(rh1))
            print(np.max(rh2), np.min(rh2))
            print(np.max(u1), np.min(u1))
            print(np.max(u2), np.min(u2))
            frame += 1
            plt.pause(0.001)


if __name__ == "__main__":
    main()
