function motionTilt(event) {
  const beta = Math.abs(Number((event && event.beta) || 0));
  const gamma = Math.abs(Number((event && event.gamma) || 0));
  return Math.max(beta, gamma);
}

function levelStateFromMotion(event) {
  const tilt = motionTilt(event);
  if (!tilt) {
    return {
      levelClass: "level-neutral",
      levelIcon: "□",
      levelHint: "请将手机放平拍摄",
    };
  }
  if (tilt > 18) {
    return {
      levelClass: "level-warning",
      levelIcon: "!",
      levelHint: "请将手机放平拍摄",
    };
  }
  if (tilt > 10) {
    return {
      levelClass: "level-caution",
      levelIcon: "·",
      levelHint: "稍微放平一点，题框会更准",
    };
  }
  return {
    levelClass: "level-good",
    levelIcon: "✓",
    levelHint: "角度可以，保持稳定",
  };
}

Page({
  data: {
    flashMode: "off",
    capturing: false,
    levelClass: "level-neutral",
    levelIcon: "□",
    levelHint: "请将手机放平拍摄",
  },

  onLoad() {
    this._emitted = false;
    this._motionHandler = (event) => {
      this.setData(levelStateFromMotion(event));
    };
    if (wx.startDeviceMotionListening && wx.onDeviceMotionChange) {
      wx.startDeviceMotionListening({ interval: "ui" });
      wx.onDeviceMotionChange(this._motionHandler);
    }
  },

  onUnload() {
    if (wx.offDeviceMotionChange && this._motionHandler) {
      wx.offDeviceMotionChange(this._motionHandler);
    }
    if (wx.stopDeviceMotionListening) {
      wx.stopDeviceMotionListening();
    }
    if (!this._emitted) {
      this.emitCameraClosed();
    }
  },

  handleClose() {
    this.emitCameraClosed();
    wx.navigateBack();
  },

  handleTakePhoto() {
    if (this.data.capturing) {
      return;
    }
    this.setData({ capturing: true });
    const camera = wx.createCameraContext();
    camera.takePhoto({
      quality: "high",
      success: (response) => {
        this.emitCapturedImage(response && response.tempImagePath);
      },
      fail: () => {
        wx.showToast({
          title: "拍照失败，请重试",
          icon: "none",
        });
      },
      complete: () => {
        this.setData({ capturing: false });
      },
    });
  },

  handleChooseAlbum() {
    if (!wx.chooseMedia) {
      wx.showToast({
        title: "当前环境不支持相册选择",
        icon: "none",
      });
      return;
    }
    wx.chooseMedia({
      count: 1,
      mediaType: ["image"],
      sourceType: ["album"],
      success: (response) => {
        const file = response.tempFiles && response.tempFiles[0];
        this.emitCapturedImage(file && file.tempFilePath);
      },
      fail: () => {},
    });
  },

  emitCapturedImage(tempFilePath) {
    if (!tempFilePath) {
      wx.showToast({
        title: "没有获取到照片",
        icon: "none",
      });
      return;
    }
    this._emitted = true;
    const channel = this.getOpenerEventChannel && this.getOpenerEventChannel();
    if (channel && channel.emit) {
      channel.emit("capturedHomeworkImage", { tempFilePath });
    }
    wx.navigateBack();
  },

  emitCameraClosed() {
    this._emitted = true;
    const channel = this.getOpenerEventChannel && this.getOpenerEventChannel();
    if (channel && channel.emit) {
      channel.emit("homeworkCameraClosed", {});
    }
  },
});
