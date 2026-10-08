# Bài phản tư - Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** CẦN_ĐIỀN_HỌ_TÊN  
**Khoá:** CẦN_ĐIỀN_KHOÁ  
**Tier đã chạy:** T4  
**Ngày:** 2026-10-08

Các con số trong bài phản tư này lấy từ các file do notebook sinh ra, chủ yếu là `adapters/dpo/dpo_metrics.json` và `data/eval/judge_summary.json`.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4 16 GB |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | TẠM_CHỜ_KẾT_QUẢ_NB2 |
| DPO: beta / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 |
| Giám khảo | `Skywork/Skywork-Reward-V2-Qwen3-4B` + `Skywork/Skywork-Reward-V2-Llama-3.2-3B` · sanity accuracy: TẠM_CHỜ_JUDGE_SUMMARY |
| Chi phí | 0 đồng, dùng Colab miễn phí |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | TẠM_CHỜ_KẾT_QUẢ_COLAB |
| VRAM cao nhất | TẠM_CHỜ_KẾT_QUẢ_COLAB |
| Reward gap cuối trên tập huấn luyện (chosen - rejected) | TẠM_CHỜ_DPO_METRICS |
| Độ chính xác reward trên held-out | TẠM_CHỜ_DPO_METRICS |
| Margin trên held-out | TẠM_CHỜ_DPO_METRICS |
| Chẩn đoán tự động (`diagnosis`) | TẠM_CHỜ_DPO_METRICS |
| Độ dài trung bình câu trả lời SFT -> DPO (NB4) | TẠM_CHỜ_JUDGE_SUMMARY |

---

## 3. Đọc đường reward (>= 100 từ)

Ảnh: `screenshots/03-dpo-reward-curves.png`

Trong NB3, tôi dùng mô hình SFT đã merge làm mô hình tham chiếu cố định, còn adapter DPO là phần được cập nhật. Theo `dpo_metrics.json`, chẩn đoán tự động là **TẠM_CHỜ_DPO_METRICS**. Reward gap cuối trên tập huấn luyện là **TẠM_CHỜ_DPO_METRICS**, còn trên held-out margin là **TẠM_CHỜ_DPO_METRICS** với độ chính xác reward **TẠM_CHỜ_DPO_METRICS**. Khi đọc biểu đồ, tôi không chỉ nhìn margin mà tách riêng hai đường `rewards/chosen` và `rewards/rejected`. Nếu `chosen` tăng còn `rejected` giảm, đây là dấu hiệu DPO hoạt động đúng kỳ vọng vì mô hình vừa tăng ưu tiên cho câu tốt hơn vừa giảm ưu tiên cho câu kém hơn. Nếu margin tăng nhưng `chosen` cũng giảm, tôi xem đây là hiện tượng likelihood displacement: mô hình vẫn phân biệt tốt hơn vì `rejected` giảm nhanh hơn, nhưng xác suất tuyệt đối của câu được chọn không tăng. Điều quan trọng là đường held-out phải đi cùng chiều với train; nếu chỉ train cải thiện còn held-out đứng yên hoặc xấu đi thì kết quả có nguy cơ là học thuộc dữ liệu preference thay vì học quy luật ưu tiên tổng quát.

Với kết quả của lần chạy này, đường train cho thấy TẠM_CHỜ_ẢNH_REWARD_CURVES. Đường held-out cho thấy TẠM_CHỜ_ẢNH_REWARD_CURVES. Vì vậy, tôi đánh giá chẩn đoán tự động là TẠM_CHỜ_DPO_METRICS. Điểm tôi lưu ý nhất là margin không tự nó chứng minh mô hình tốt hơn trong hội thoại thực tế; nó chỉ chứng minh mô hình đang tối ưu đúng objective preference trên dữ liệu đã cho. Kết luận cuối cùng vẫn cần đối chiếu với NB4, đặc biệt là win rate, khoảng tin cậy, sanity accuracy của giám khảo và ảnh hưởng của độ dài câu trả lời.

---

## 4. So sánh SFT vs SFT+DPO

Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | TẠM_CHỜ_JUDGE_SUMMARY | TẠM_CHỜ_JUDGE_SUMMARY | TẠM_CHỜ_JUDGE_SUMMARY | TẠM_CHỜ_JUDGE_SUMMARY | TẠM_CHỜ_JUDGE_SUMMARY | TẠM_CHỜ_JUDGE_SUMMARY | TẠM_CHỜ_JUDGE_SUMMARY |
| hữu ích - helpfulness (4) | 4 | TẠM_CHỜ_JUDGE_SUMMARY | TẠM_CHỜ_JUDGE_SUMMARY | TẠM_CHỜ_JUDGE_SUMMARY | TẠM_CHỜ_JUDGE_SUMMARY | TẠM_CHỜ_JUDGE_SUMMARY | TẠM_CHỜ_JUDGE_SUMMARY |
| an toàn - safety (4) | 4 | TẠM_CHỜ_JUDGE_SUMMARY | TẠM_CHỜ_JUDGE_SUMMARY | TẠM_CHỜ_JUDGE_SUMMARY | TẠM_CHỜ_JUDGE_SUMMARY | TẠM_CHỜ_JUDGE_SUMMARY | TẠM_CHỜ_JUDGE_SUMMARY |

Giám khảo: `Skywork/Skywork-Reward-V2-Qwen3-4B` + `Skywork/Skywork-Reward-V2-Llama-3.2-3B`. Sanity accuracy: TẠM_CHỜ_JUDGE_SUMMARY. `score_length_spearman` hoặc position consistency: TẠM_CHỜ_JUDGE_SUMMARY.

Kết quả NB4 cho thấy DPO có win rate held-out là **TẠM_CHỜ_JUDGE_SUMMARY** với khoảng tin cậy 95% là **TẠM_CHỜ_JUDGE_SUMMARY**. Nếu khoảng này chứa 0.5 thì tôi không kết luận chắc chắn DPO tốt hơn SFT, mà chỉ nói rằng lần chạy này chưa đủ bằng chứng thống kê để phân biệt hai mô hình. Nếu khoảng này nằm hẳn trên 0.5 thì có thể nói DPO cải thiện theo tiêu chí của giám khảo, nhưng vẫn cần kiểm tra độ tin cậy của giám khảo. Sanity accuracy là **TẠM_CHỜ_JUDGE_SUMMARY**; nếu dưới 0.8, tôi sẽ xem win rate là tín hiệu yếu vì giám khảo chưa đọc tốt các cặp kiểm tra tiếng Việt.

Một điểm quan trọng là thiên vị độ dài. Tỉ lệ câu dài hơn thắng là **TẠM_CHỜ_JUDGE_SUMMARY**, còn win rate trên các cặp dài gần bằng nhau là **TẠM_CHỜ_JUDGE_SUMMARY**. Nếu DPO thắng cao nhưng câu dài hơn gần như luôn thắng, có khả năng DPO đang học cách trả lời dài hơn thay vì thật sự hữu ích hơn. Khi đọc `per_judge`, tôi cũng so sánh win rate giữa từng giám khảo. Nếu một giám khảo cho DPO thắng cao hơn hẳn giám khảo còn lại, tôi xem đó là dấu hiệu cần thận trọng vì có thể tồn tại preference leakage hoặc khác biệt thiên kiến giữa các reward model.

Ví dụ về độ hữu ích: với prompt **TẠM_CHỜ_SIDE_BY_SIDE_JSONL**, câu trả lời của **TẠM_CHỜ_SIDE_BY_SIDE_JSONL** tốt hơn vì TẠM_CHỜ_SIDE_BY_SIDE_JSONL. Ví dụ về an toàn: với prompt **TẠM_CHỜ_SIDE_BY_SIDE_JSONL**, câu trả lời của **TẠM_CHỜ_SIDE_BY_SIDE_JSONL** phù hợp hơn vì TẠM_CHỜ_SIDE_BY_SIDE_JSONL. Hai ví dụ này giúp kiểm tra win rate bằng mắt thường, tránh chỉ dựa vào một con số tổng hợp.

---

## 5. Đánh đổi theo beta (bonus `make beta-sweep`)

| beta | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | Không chạy | Không chạy | Không chạy | Beta nhỏ thường cho phép mô hình đi xa reference hơn, có thể tăng margin nhưng cũng dễ làm câu trả lời lệch hơn. |
| 0.1 | Không chạy | Không chạy | Không chạy | Đây là cấu hình chính của lab, cân bằng giữa học preference và giữ mô hình gần SFT. |
| 0.5 | Không chạy | Không chạy | Không chạy | Beta lớn thường ràng buộc mạnh hơn với reference, nên thay đổi có thể chậm và margin tăng ít hơn. |

Tôi không chạy bonus beta-sweep. Dự đoán của tôi là beta nhỏ hơn sẽ làm reward margin thay đổi mạnh hơn nhưng có rủi ro tăng thiên vị độ dài hoặc làm chất lượng sinh không ổn định. Beta lớn hơn sẽ bảo thủ hơn, có thể giữ văn phong SFT tốt hơn nhưng hiệu ứng preference yếu hơn.

---

## 6. Một quyết định quan trọng nhất (>= 150 từ)

Quyết định quan trọng nhất của tôi là dùng cấu hình DPO mặc định của lab trên Colab T4, đặc biệt là giữ beta ở **0.1** và tốc độ học **5e-6**, thay vì tăng tốc độ học hoặc giảm mạnh lượng dữ liệu để chạy nhanh hơn. Phương án thay thế là giảm `PREF_TRAIN`, giảm `MAX_LEN`, hoặc đổi beta để rút ngắn thời gian, nhưng các lựa chọn đó có thể làm kết quả khó so sánh với rubric. Vì mục tiêu của bài này không chỉ là chạy cho xong mà còn phải đọc reward curve, so train với held-out và phân tích win rate, tôi ưu tiên cấu hình chuẩn để các số liệu có ý nghĩa hơn.

Kết quả sau khi chạy cho thấy **TẠM_CHỜ_KẾT_QUẢ_COLAB**. Điều này TẠM_CHỜ_NHẬN_XÉT_CÁ_NHÂN so với kỳ vọng ban đầu của tôi. Nếu làm lại, tôi sẽ TẠM_CHỜ_NHẬN_XÉT_CÁ_NHÂN. Tôi cũng sẽ theo dõi kỹ hơn tỉ lệ câu dài hơn thắng, vì DPO có thể tối ưu preference bằng cách viết dài hơn thay vì trả lời đúng trọng tâm hơn. Với một bài lab alignment nhỏ, tôi thấy phần đáng tin nhất không phải là một win rate đơn lẻ, mà là sự nhất quán giữa ba nguồn bằng chứng: reward curve trên held-out, sanity accuracy của giám khảo và ví dụ so sánh side-by-side. Khi ba tín hiệu này cùng hướng, kết luận về tác dụng của DPO thuyết phục hơn nhiều.

---

## 7. Bộ đo chuẩn (bonus NB6, >= 150 từ)

Không chạy bonus NB6.

| Bộ đo | Giới hạn / môn con | SFT (+/- stderr) | SFT+DPO (+/- stderr) | Delta |
|---|---:|---:|---:|---:|
| IFEval | Không chạy | Không chạy | Không chạy | Không chạy |
| GSM8K | Không chạy | Không chạy | Không chạy | Không chạy |
| Global-MMLU-vi | Không chạy | Không chạy | Không chạy | Không chạy |

Vì không chạy benchmark, tôi không kết luận về alignment tax trên GSM8K hoặc khả năng làm theo chỉ dẫn theo IFEval. Nếu có thêm thời gian, tôi sẽ chạy NB6 để kiểm tra xem cải thiện preference trong NB4 có đi kèm suy giảm năng lực lập luận hay kiến thức hay không.

---

## 8. Biến thể loss (bonus NB3b)

Không chạy bonus NB3b.

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | Không chạy | Không chạy | Không chạy | Không chạy |
| RPO | Không chạy | Không chạy | Không chạy | Không chạy |
| DPO-norm | Không chạy | Không chạy | Không chạy | Không chạy |
| LD-DPO | Không chạy | Không chạy | Không chạy | Không chạy |
| ORPO | Không chạy | Không chạy | Không chạy | Không chạy |

---

## 9. GRPO (bonus NB7)

Không chạy bonus NB7.

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | Không chạy |
| Sai số chuẩn xấp xỉ sqrt(p(1-p)/n) | Không chạy |

---

## Danh sách bonus

- [ ] NB3b - biến thể loss (+8)
- [ ] NB5 - GGUF SFT+DPO (+4)
- [ ] NB6 - benchmark (+6)
- [ ] NB7 - GRPO (+8)
- [ ] beta-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

TẠM_CHỜ_KẾT_QUẢ_COLAB
