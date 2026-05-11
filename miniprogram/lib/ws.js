function createDebugSocket(options) {
  const socketTask = wx.connectSocket({
    url: options.url,
  });

  let opened = false;
  let closed = false;
  let settleOpen;
  let rejectOpen;

  const ready = new Promise((resolve, reject) => {
    settleOpen = resolve;
    rejectOpen = reject;
  });

  function failBeforeOpen(error) {
    if (!opened && rejectOpen) {
      rejectOpen(error);
      rejectOpen = null;
      settleOpen = null;
    }
  }

  socketTask.onOpen(() => {
    opened = true;
    if (settleOpen) {
      settleOpen();
      settleOpen = null;
      rejectOpen = null;
    }
    if (options.onOpen) {
      options.onOpen();
    }
  });

  socketTask.onMessage((response) => {
    try {
      const payload = JSON.parse(response.data);
      if (options.onEvent) {
        options.onEvent(payload);
      }
    } catch (_error) {
      if (options.onError) {
        options.onError(new Error("WebSocket returned invalid JSON."));
      }
    }
  });

  socketTask.onError((response) => {
    const error = new Error(
      (response && response.errMsg) || "WebSocket connection failed."
    );
    failBeforeOpen(error);
    if (options.onError) {
      options.onError(error);
    }
  });

  socketTask.onClose((response) => {
    if (!opened) {
      failBeforeOpen(new Error("WebSocket closed before the handshake completed."));
    }
    if (closed) {
      return;
    }
    closed = true;
    if (options.onClose) {
      options.onClose(response);
    }
  });

  function sendJSON(payload) {
    return ready.then(
      () =>
        new Promise((resolve, reject) => {
          socketTask.send({
            data: JSON.stringify(payload),
            success: resolve,
            fail: (error) => {
              reject(new Error((error && error.errMsg) || "WebSocket send failed."));
            },
          });
        })
    );
  }

  function close() {
    if (closed) {
      return;
    }
    closed = true;
    try {
      socketTask.close({});
    } catch (_error) {
      // Ignore close errors during page teardown.
    }
  }

  return {
    ready,
    sendJSON,
    close,
  };
}

module.exports = {
  createDebugSocket,
};
