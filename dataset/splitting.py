import random

import tensorflow as tf


"""
五折交叉验证划分训练集和测试集病人的id。
"""

def get_train_test_sets_patients_id(mode):
    assert mode in ['random', 'split1', 'split2', 'split3', 'split4', 'split5']
    all_ids = [122869, 136016, 150530, 154076, 175524,241402, 252657, 255976, 280082, 284058,
                  314592, 329086, 351570, 373887, 389673,403480, 416383, 417172, 420410, 423879,
                  425726, 445956, 464303, 471442, 473285,474799, 478708, 479158, 480081, 480678,
                  481808, 482347, 482973, 483647, 484652,489020, 494030, 497652, 497862, 500074,
                  500302, 500765, 501194, 501273, 504651,504881, 505844, 507019, 508825, 513189,
                  516934, 517902, 518208, 518659, 520248,523600, 523902, 524453, 524602, 546027,
                  562240, 566226, 566775, 568795, 571458,573894, 575752, 579205, 579454, 579502,
                  579784, 579918, 584802, 585066,

               129764, 135769, 143739, 161363, 171460, 182176, 202577, 226373, 227388, 234045,
               238442, 249437, 253312, 258370, 276536, 286557, 292990, 297917, 301549, 314638,
               316873, 337247, 351504, 353645, 368630, 376451, 378324, 378836, 389941, 390992,
               410725, 417998, 420962, 421146, 427066, 430109, 441292, 444269, 465951, 466823,
               474962, 475856, 477177, 478338, 479469, 480411, 480942, 483431, 484932, 485632,
               488393, 494269, 495301, 495663, 496566, 496855, 497388, 497394, 498420, 500034,
               501290, 504566, 505885, 506002, 507351, 509387, 511069, 512189, 520165, 520408,
               522190, 523243, 525857, 526401, 527776, 534201, 534524, 535039, 536082, 536296,
               542240, 542761, 543446, 543573, 547078, 547291, 549616, 565327, 565752, 566093,
               567692, 567701, 568324, 570072, 570300, 572294, 572996, 574848, 575351, 576366,
               577119, 577300, 580502, 583413, 584236, 586834, 588351, 588637, 589026, 589939
               ]
    if mode == 'random':
        # 取两份作为测试集，其余为训练集
        ids_test = []
        begnign =[122869, 136016, 150530, 154076, 175524,241402, 252657, 255976, 280082, 284058,
                  314592, 329086, 351570, 373887, 389673,403480, 416383, 417172, 420410, 423879,
                  425726, 445956, 464303, 471442, 473285,474799, 478708, 479158, 480081, 480678,
                  481808, 482347, 482973, 483647, 484652,489020, 494030, 497652, 497862, 500074,
                  500302, 500765, 501194, 501273, 504651,504881, 505844, 507019, 508825, 513189,
                  516934, 517902, 518208, 518659, 520248,523600, 523902, 524453, 524602, 546027,
                  562240, 566226, 566775, 568795, 571458,573894, 575752, 579205, 579454, 579502,
                  579784, 579918, 584802, 585066]
        keys = random.sample(range(0, 74), 15)
        for key in keys:
            ids_test.append(begnign[key])
        malignant = [129764, 135769, 143739, 161363, 171460, 182176, 202577, 226373, 227388, 234045,
                     238442, 249437, 253312, 258370, 276536, 286557, 292990, 297917, 301549, 314638,
                     316873, 337247, 351504, 353645, 368630, 376451, 378324, 378836, 389941, 390992,
                     410725, 417998, 420962, 421146, 427066, 430109, 441292, 444269, 465951, 466823,
                     474962, 475856, 477177, 478338, 479469, 480411, 480942, 483431, 484932, 485632,
                     488393, 494269, 495301, 495663, 496566, 496855, 497388, 497394, 498420, 500034,
                     501290, 504566, 505885, 506002, 507351, 509387, 511069, 512189, 520165, 520408,
                     522190, 523243, 525857, 526401, 527776, 534201, 534524, 535039, 536082, 536296,
                     542240, 542761, 543446, 543573, 547078, 547291, 549616, 565327, 565752, 566093,
                     567692, 567701, 568324, 570072, 570300, 572294, 572996, 574848, 575351, 576366,
                     577119, 577300, 580502, 583413, 584236, 586834, 588351, 588637, 589026, 589939]
        keys = random.sample(range(0, 110), 22)
        for key in keys:
            ids_test.append(malignant[key])
        ids_train = list(set(all_ids).difference(set(ids_test)))
    elif mode == 'split1':
        ids_test = [122869, 136016, 150530, 154076, 175524,
                    241402, 252657, 255976, 280082, 284058,
                    314592, 329086, 351570, 373887, 389673,

                    129764, 135769, 143739, 161363, 171460,
                    182176, 202577, 226373, 227388, 234045,
                    238442, 249437, 253312, 258370, 276536,
                    286557, 292990, 297917, 301549, 314638,
                    316873, 337247,
                    ]
        ids_train = list(set(all_ids).difference(set(ids_test)))
    elif mode == 'split2':
        ids_test = [403480, 416383, 417172, 420410,423879,
                    425726, 445956, 464303, 471442,473285,
                    474799, 478708, 479158, 480081, 480678,

                    351504, 353645, 368630, 376451, 378324,
                    378836, 389941, 390992, 410725, 417998,
                    420962, 421146, 427066, 430109, 441292,
                    444269, 465951, 466823, 474962, 475856,
                    477177, 478338,

                    ]
        ids_train = list(set(all_ids).difference(set(ids_test)))  # 求差集。
    elif mode == 'split3':
        ids_test = [481808, 482347, 482973, 483647, 484652,
                    489020, 494030, 497652, 497862, 500074,
                    500302, 500765, 501194, 501273, 504651,

                    479469, 480411, 480942, 483431, 484932,
                    485632, 488393, 494269, 495301, 495663,
                    496566, 496855, 497388, 497394, 498420,
                    500034, 501290, 504566, 505885, 506002,
                    507351, 509387,

                    ]
        ids_train = list(set(all_ids).difference(set(ids_test)))  # 求差集。
    elif mode == 'split4':
        ids_test = [504881, 505844, 507019, 508825, 513189,
                    516934, 517902, 518208, 518659, 520248,
                    523600, 523902, 524453, 524602, 546027,

                    511069, 512189, 520165, 520408, 522190,
                    523243, 525857, 526401, 527776, 534201,
                    534524, 535039, 536082, 536296, 542240,
                    542761, 543446, 543573, 547078, 547291,
                    549616, 565327,

                    ]
        ids_train = list(set(all_ids).difference(set(ids_test)))  # 求差集。
    else:
        ids_test = [562240, 566226, 566775, 568795, 571458,
                    573894, 575752, 579205, 579454, 579502,
                    579784, 579918, 584802, 585066,

                    565752, 566093, 567692, 567701, 568324,
                    570072, 570300, 572294, 572996, 574848,
                    575351, 576366, 577119, 577300, 580502,
                    583413, 584236, 586834, 588351, 588637,
                    589026, 589939
                    ]
        ids_train = list(set(all_ids).difference(set(ids_test)))  # 求差集。
    return ids_train, ids_test

def main(_):
    dataset_train_ids, dataset_test_ids = get_train_test_sets_patients_id('split1')
    print('dataset_train_ids:', dataset_train_ids)
    print('dataset_test_ids:', dataset_test_ids)

if __name__ == '__main__':
    tf.app.run()