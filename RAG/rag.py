import threading
import time

def update_vector_index():
    print("[New Thred] Updating vector store")
    time.sleep(5)
    print("[New Thread] Update completed")


def loading(stop_event):
    animation = ["-", "/", "-", "\\"]
    idx = 0
    while not stop_event.is_set():
        print(animation[idx % len(animation)], end = "\r")
        idx += 1
        time.sleep(0.2)

def check_query(query : str):
    if "__update__store__" in query:
        print("Keyword detected")
    
    # control the ascii animation
    stop_event = threading.Event()

    # new thread to update vector store
    update_thread = threading.Thread(target = update_vector_index)
    update_thread.start()

    # ascii thread
    ascii_thread = threading.Thread(target=loading, args = (stop_event, ))
    ascii_thread.start()

    # waiting for update_thread to finish indexing
    update_thread.join()

    # stop ascii animation
    stop_event.set()
    ascii_thread.join()

    print("Continuing on the main thread")

if __name__ == "__main__":
    check_query("__update__store__ this is a test")