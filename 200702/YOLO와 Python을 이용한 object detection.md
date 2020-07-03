## YOLO와 Python을 이용한 object detection
[블로그](https://reyrei.tistory.com/16) 참고
- yolov2 사용
- cuda v9.0 (9버전 사용하긴 했는데 지금 사용하는 노트북의 nvidia랑 잘 맞지않는다 . . . ㅠㅠ . . gpu로 돌리도록 옵션을 줬을 때 gpu mode라고 뜨긴하는데 . . 흠)
- cuDNN v7.0.3 (cuda v9.0과 맞는 버전 다운로드 함)

### cpu 사용시
![KakaoTalk_20200702_095952497](https://user-images.githubusercontent.com/59993071/86421019-5ff2d700-bd13-11ea-9576-b8af1928b0a9.png)

### gpu 사용시
![KakaoTalk_20200702_101144115](https://user-images.githubusercontent.com/59993071/86421033-7436d400-bd13-11ea-8e3a-c9dc4421f86f.png)
- 근데 에러는 아니지만 cuda v10의 dll 파일들을 필요로 하는 메세지가 뜬다 ㅠ 담주에 학원에서 새로운 노트북을 준다고 하니 그 노트북에서 한 번 다시 해봐야할듯함 ㅠ
