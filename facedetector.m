clc;                            % basic statement
clear variables;
close all;
                    
load('wiki.mat');           %load database file
gender = wiki.gender;         %difine gender
full_path = wiki.full_path;     %Given path
load('new_gender.mat');
load('path.mat')
% faceDetector=vision.CascadeObjectDetector('FrontalFaceCART'); %Create a detector object
k=1;

index = [9,14,15,26,31,47,67,68,65,64,73,74,79,87,118,120,121,123,129,133,139,146,151,152,153,154,155,...
    172,175,183,185,188,196,197,199,207,211,218,219,233,234,248,252,265,269,271,272,292,303,304,312,313,333,...
    337,345,350,352,355,356,358,360,362,378,381,385,399,405,421,432,439,440,453,456,458,461,462,464,466,468,...
    469,470,473,474,476,477,483,491,492,493,494,497,501,504,507,508,512,526,557,558,560,565,568,633,640,664,...
    674,675,677,679,680,681,685,686,688,697,702,703,704,706,711,714,716,717,720,721,724,725,726,728,735,736,...
    739,759,761,762,763,768,770,771,779,805,821,837,839,850,873,883,884,887,911,923,980,988,990,999,1000,...
    1001,1010,1012,1022,1025,1029,1032,1033,1035,1044,1053,...
    1054,1057,1080,1091,1095,1107,1119,1122,1124,1125,1128,1133,1137,1141,1144,1145,1149,1150,1168,1184,1202,...
    1203,1209,1213,1223,1235,1242,1250,1258,1263,1268,1282,1286,1305,1306,1314,1318,1326,1327,1329,1342,1344,...
    1352,1358,1359,1363,1368,1369,1372,1381,1380,1388,1390,1401,1402,1403,1409,1414,1423,1424,1427,1429,1433,...
    1435,1436,1437,1438,1439,1441,1442,1443,1446,1449,1451,1460,1463,1487,1496,1502,1503,1505,1519,1531,1535,...
    1553,1563,1565,1568,1569,1570,1578,1595,1604,1613,1615,1616,1624,1635,1636,1641,1642,1650,1651,1655,1657,...
    1659,1671,1672,1673,1674,1675,1676,1677,1679,1680,1681,1682,1693,1694,1695,1706,1717,1719,1722,1724,1725,...
    1734,1739,1745,1747,1751,1752,1758,1759,1760,1761,1765,1771,1774,1778,1785,1812,1822,1831,1846,1852,1858,...
    1864,1867,1870,1886,1892,1894,1916,1932,1936,1938,1942,1944,1948,1952,1966,1984,2011,2015,2016,2024,2041,...
    2043,2073,2079,2099,2125,2127,2139,2170,2174,2176,2182,2188,2206,2208,2220,2221,2228,2240,2242,2252,2258,...
    2260,2287,2289,2292,2300,2314,2316,2320,2335,2371,2379,2385,2387,2389,2395,2401,2402,2405,2422,2423,2425,...
    2435,2442,2443,2452,2456,2488,2490,2493,2494,2499,2522,2525,2527,2530,2533,2537,2542,2543,2544,2549,2552,...
    2560,2563,2577,2592,2594,2617,2630,2644,2645,2660,2668,2683,2692,2697,2704,2706,2707,2709,2711,2731,2758,...
    2765,2767,2771,2773,2774,2779,2782,2786,2789,2793,2798,2802,2811,2813,2822,2823,2826,2832,2833,2840,2850,...
    2860,2867,2882,2894,2900,2901,2908,2909,2916,2923,2925,2941,2967,2971,2972,2975,2992,2995,2996,2998,3006,...
    3024,3028,3029,3033,3034,3041,3042,3047,3048,3058,3074,3075,3076,3077,3079,3086,3088,3097,3099,3112,3114,...
    3115,3125,3140,3142,3143,3147,3149,3151,3152,3158,3161,3163,3164,3165,3166,3168,3181,3183,3191,3206,3219,...
    3231,3235,3264,3265,3266,3276,3281,3294,3297,3301,3302,3303,3305,3314,3322,3330,3346,3350,3369,3383,3385,...
    3389,3391,3393,3396,3397,3415,3417,3428,3431,3437,3447,3457,3460,3461,3462,3516,3521,3542,3552,3571,3576,...
    3581,3591,3597,3602,3603,3609,3614,3629,3642,3658,3659,3665,3673,3674,3675,3690,3691,3711,3713,3714,3718,...
    3723,3737,3738,3740,3742,3743,3752,3756,3763,3764,3766,3767,3772,3776,3777,3781,3789,3790,3791,3792,3795,...
    3796,3808,3814,3816,3817,3833,3834,3838,3840,3843,3853,3860,3867,3883,3894,3916,3920,3939,3942,3943,3949,...
    3967,3977,3979,3984,3985,3988,4002,4005,4006,4012,4019,4035,4040,4043,4044,4045,4046,4059,4064,4067,4078,...
    4091,4118,4125,4128,4131,4136,4137,4138,4143,4156,4158,4163,4175,4180,4185,4187,4193,4203,4212,4219,4222,...
    4245,4254,4256,4258,4260,4265,4267,4271,4288,4298,4315,4321,4322,4325,4326,4329,4334,4342,4352,4377,4378,...
    4398,4406,4419,4427,4428,4443,4444,4448,4468,4469,4471,4472,4476,4477,4482,4493,4498,4507,4534,4536,4545,...
    4547,4558,4569,4571,4590,4595,4597,4601,4603,4604,4622,4628,4642,4644,4645,4654,4658,4660,4661,4669,4673,...
    4677,4682,4685,4689,4690,4691,4699,4700,4716,4717,4718,4722,4724,4725,4732,4733,4734,4735,4736,4739,4740,...
    4754,4756,4758,4762,4763,4768,4769,4770,4771,4772,4773,4776,4781,4782,4785,4787,4788,4789,4797,4798,4799,...
    4800,4801,4802,4804,4808,4815,4816,4818,4823,4828,4835,4836,4841,4846,4847,4853,4855,4858,4886,4898,4901,...
    4907,4944,4950,4952,4970,5034,5035,5045,5046,5054,5061,5071,5073,5079,5082,5084,5097,5109,5110,5127,5130,...
    5131,5141,5142,5143,5145,5159,5176,5179,5180,5185,5196,5202,5207,5210,5211,5212,5213,5214,5220,5222,5225,...
    5228,5230,5231,5234,5241,5244,5245,5249,5257,5259,5260,5262,5264,5271,5280,5283,5286,5287,5304,5315,5318,...
    5322,5327,5330,5341,5355,5377,5379,5389,5391,5397,5401,5402,5403,5420,5432,5457,5458,5475,5476,5480,5490,...
    5492,5494,5505,5511,5512,5517,5522,5548,5557,5559,5562,5573,5580,5598,5622,5627,5629,5639,5656,5659,5667,...
    5680,5689,5700,5702,5703,5706,5708,5711,5714,5715,5718,5718,5725,5733,5734,5742,5744,5752,5761,5762,5766,...
    5771,5772,5777,5782,5794,5802,5813,5819,5820,5828,5830,5832,5834,5834,5836,5847,5848,5856,5865,5867,5869,...
    5883,5886,5888,5890,5891,5893,5895,5900,5901,5904,5905,5912,5918,5920,5926,5930,5939,5954,5988,6008,6014,...
    6016,6017,6025,6039,6045,6051,6056,6057,6066,6069,6070,6077,6081,6083,6084,6089,6096,6097,6100,6104,6105,...
    6108,6123,6126,6132,6140,6148,6150,6155,6176,6177,6179,6193,6223,6224,6230,6234,6236,6239,6241,6243,6247,...
    6258,6274,6279,6281,6282,6290,6292,6296,6303,6310,6312,6315,6316,6326,6330,6337,6338,6346,6357,6360,6365,...
    6372,6374,6375,6386,6388,6391,6407,6414,6422,6423,6434,6435,6442,6448,6452,6455,6457,6460,6462,6474,6475,...
    6482,6487,6487,6489,6490,6508,6515,6516,6521,6522,6526,6537,6568,6543,6553,6555,6556,6561,6567,6570,6571,...
    6577,6582,6583,6586,6587,6590,6593,6598,6613,6619,6630,6647,6650,6654,6656,6657,6666,6690,6697,6698,6707,...
    6708,6753,6763,6767,6784,6790,6792,6796,6798,6805,6819,6820,6821,6827,6849,6865,6867,6873,6876,6886,6896,...
    6903,6904,6933,6934,6935,6936,6940,6954,6955,6956,6975,6986,6992,7002,7007,7012,7013,7014,7015,7018,7019,...
    7022,7023,7029,7030,7031,7032,7034,7036,7037,7039,7041,7047,7058,7059,7066,7068,7071,7072,7076,7078,7080,...
    7081,7083,7084,7085,7086,7088,7091,7092,7095,7098,7110,7111,7112,7115,7116,7117,7120,7126,7127,7129,7136,...
    7137,7139,7140,7144,7145,7146,7148,7149,7154,7155,7164,7165,7167,7173,7174,7176,7182,7206,7208,7217,7222,...
    7226,7230,7231,7231,7232,7235,7236,7238,7266,7272,7280,7304,7313,7314,7315,7319,7328,7341,7344,7350,7364,...
    7365,7368,7372,7381,7394,7395,7396,7397,7399,7400,7401,7402,7403,7408,7410,7411,7413,7414,7420,7439,7441,...
    7443,7446,7462,7466,7474,7475,7476,7477,7478,7479,7481,7483,7485,7486,7487,7491,7500,7502,7504,7513,7519,...
    7520,7522,7538,7541,7554,7564,7568,7577,7586,7588,7590,7610,7611,7612,7614,7620,7621,7622,7633,7637,7644,...
    7651,7652,7653,7655,7663,7668,7670,7671,7672,7673,7684,7687,7699,7700,7704,7706,7707,7714,7716,7718,7723,...
    7724,7726,7728,7729,7736,7745,7747,7749,7751,7753,7756,7758,7763,7764,7773,7774,7778,7782,7787,7788,7790,...
    7795,7799,7801,7808,7815,7824,7836,7838,7842,7850,7853,7861,7863,7879,7882,7887,7888,7898,7911,7917,7954,...
    7959,7972,7974,7981,7986,7989,7994,8032,8042,8043,8053,8068,8071,8075,8085,8097,8105,8107,8109,8111,8114,...
    8119,8123,8133,8135,8145,8146,8154,8160,8161,8171,8177,8179,8180,8183,8186,8191,8204,8208,8226,8231,8240,...
    8246,8251,8272,8274,8277,8278,8283,8284,8285,8285,8285,8287,8290,8292,8294,8295,8307,8308,8309,8310,8311,...
    8312,8316,8320,8322,8323,8328,8332,8342,8343,8346,8365,8366,8372,8373,8376,8376,8386,8387,8388,8390,8396,...
    8405,8407,8412,8416,8424,8426,8427,8438,8444,8448,8457,8474,8474,8478,8479,8486,8492,8496,8501,8502,8517,...
    8522,8523,8524,8532,8538,8541,8548,8552,8564,8565,8585,8608,8640,8642,8644,8647,8650,8653,8655,8666,8668,...
    8672,8680,8685,8689,8700,8702,8703,8709,8710,8716,8727,8728,8736,8746,8781,8787,8790,8802,8806,8810,8811,...
    8832,8839,8840,8845,8847,8850,8856,8858,8866,8870,8875,8879,8880,8882,8891,8892,8903,8906,8910,8913,8914,...
    8923,8944,8948,8949,8950,8954,8956,8959,8962,8975,8976,8980,8985,8986,8989,8990,8992,8999,9000,9003,9009,...
    9010,9014,9021,9022,9023,9035,9038,9041,9067,9076,9077,9101,9107,9113,9120,9125,9139,9154,9162,9164,9165,...
    9172,9179,9183,9184,9185,9187,9191,9195,9214,9217,9218,9247,9260,9262,9265,9266,9267,9273,9277,9282,9283,...
    9296,9301,9309,9311,9320,9321,9326,9331,9342,9381,9382,9386,9396,9397,9407,9416,9417,9427,9431,9439,9440,...
    9446,9454,9463,9469,9471,9486,9486,9488,9490,9492,9496,9506,9507,9512,9516,9519,9523,9526,9527,9528,9532,...
    9544,9545,9556,9557,9559,9564,9578,9579,9583,9584,9585,9591,9593,9596,9598,9599,9601,9604,9606,9608,9620,...
    9621,9628,9630,9640,9646,9647,9668,9674,9688,9691,9694,9697,9719,9723,9744,9747,9749,9766,9767,9768,9769,...
    9770,9771,9772,9773,9774,9775,9778,9779,9780,9781,9783,9791,9795,9799,9803,9812,9813,9814,9817,9834,9835,...
    9843,9847,9849,9856,9868,9870,9876,9882,9887,9898,9899,9911,9923,9932,9941,9944,9960,9982,9992,9997,10001,...
    10002,10006,10008,10014,10015,10016,10021,10022,10028,10029,10033,10034,10037,10039,10040,10043,10045,10046,...
    10048,10056,10059,10060,10070,10071,10074,10075,10077,10082,10084,10086,10087,10089,10090,10092,10093,10096,...
    10098,10103,10107,10111,10113,10114,10115,10117,10120,10129,10131,10132,10134,10135,10140,10169,10193,10195,...
    10203,10206,10207,10215,10241,10243,10244,10250,10255,10262,10283,10295,10300,10307,10319,10320,10322,10348,...
    10350,10351,10354,10359,10365,10373,10375,10383,10385,10390,10395,10396,10407,10408,10409,10411,10420,10422,...
    10426,10433,10433,10435,10436,10438,10440,10441,10443,10444,10446,10449,10450,10451,10453,10461,10475,10481,...
    10491,10493,10497,10503,10504,10527,10531,10541,10545,10548,10554,10556,10564,10569,10580,10586,10596,10597,...
    10601,10603,10609,10611,10619,10626,10629,10632,10644,10645,10646,10649,10651,10652,10653,10654,10655,10666,...
    10669,10670,10674,10675,10682,10695,10697,10702,10706,10708,10712,10714,10715,10722,10728,10730,10731,10733,...
    10739,10745,10746,10747,10767,10770,10781,10799,10814,10829,10831,10841,10843,10855,10864,10871,10881,10884,...
    10885,10894,10898,10909,10914,10907,10922,10941,10943,10959,10963,10977,10985,10990,10996,10997,11007,11010,...
    11011,11014,11015,11016,11017,11018,11019,11020,11021,11028,11038,11039,11049,11068,11069,11072,11075,11077,...
    11078,11085,11090,11103,11107,1118,11115,11121,11124,11145,11151,11152,11153,11159,11164,11166,11167,11169,...
    11170,11175,11177,11179,11182,11190,11194,11202,11218,11238,11233,11235,11236,11263,11264,11290,11297,13000,...
    11322,11329,11332,11350,11353,11356,11361,11380,11382,11390,11391,11395,11402,11409,11410,11424,11441,1145,...
    11481,11484,11485,11486,11491,11492,11493,11495,11504,11507,11510,11513,11527,11528,11529,11531,11541,11547,...
    11555,11578,11580,11588,11589,11590,11591,11595,11600,11601,11602,11610,11614,11616,11617,11621,11622,11623,...
    11629,11631,11639,11640,11642,11643,11644,11645,11647,11649,11660,11661,11667,11669,11684,11689,11690,11694,...
    11696,11711,11714,11727,11732,11734,11760,11769,11780,11811,11819,11864,11870,11884,11885,11895,11896,11901,...
    11921,11930,11931,11932,11934,11937,11938,11940,11949,11961,11982,11983,11984,11987,11988,12004,12012,12013,...
    12029,12035,12042,12050,12054,12059,12063,12069,12085,12110,12124,12136,12137,12140,12144,12151,12153,12158,...
    12187,12192,12195,12201,12202,12203,12208,12209,12211,12234,12240,12255,12261,12275,12277,12279,12280,12302,...
    12305,12310,12313,12316,12319,12324,12328,12329,12331,12332,12334,12338,12346,12347,12348,12351,12352,12355,...
    12366,12373,12378,12382,12383,12384,12388,12393,12408,12411,12422,12429,12433,12436,12443,12450,12463,12466,...
    12471,12472,12474,12476,12478,12491,12499,12501,12503,12513,12527,12557,12558,12561,12562,12564,12565,12566,...
    12567,12568,12572,12575,12587,12589,12603,12606,12608,12620,12624,12627,12639,12642,12654,12660,12662,12664,...
    12679,12680,12682,12685,12692,12696,12698,12707,12708,12710,12714,12715,12716,12722,12725,12728,12733,12744,...
    12751,12756,12762,12767,12775,12788,12790,12794,12806,12808,12816,12818,12820,12822,12835,12837,12839,12841,...
    12843,12844,12845,12846,12847,12850,12853,12855,12865,12871,12874,12876,12881,12886,12889,12891,12892,12894,...
    12896,12903,12904,12905,12910,12913,12914,12916,12917,12918,12931,12934,12934,12935,12936,12937,12946,12957,...
    12959,12961,12963,12968,12976,12985,12991,12993,12997,13001,13007,13015,13017,13025,13028,13029,13030,13032,...
    13033,13043,13044,13049,13052,13054,13055,13056,13057,13060,13061,13071,13073,13074,13080,13081,13095,13106,...
    13107,13107,13111,13131,13133,13143,13144,13145,13151,13159,13174,13176,13177,13178,13179,13180,13199,13206,...
    13209,13221,13232,13234,13236,13237,13240,13247,13248,13258,13266,13273,13276,13280,13282,13289,13292,13293,...
    13298,13304,1306,13315,13317,13318,13324,13333,13351,13359,13360,13361,13362,13362,13364,13380,13383,13385,...
    13386,13387,13416,13417,13418,13429,13434,13441,13451,13453,13455,13461,13462,13463,13464,13466,13468,13470,...
    13471,13472,13476,13477,13490,13504,13506,13510,13518,13519,13524,13525,13527,13530,13536,13537,13538,13545,...
    13546,13554,13555,13556,13564,13568,13578,13579,13582,13585,13588,13590,13596,13597,13598,13599,13602,13605,...
    13609,13611,13614,13618,13619,13623,13632,13636,13638,13649,13655,13660,13664,13665,13672,13705,13713,13719,...
    13724,13729,13730,13732,13733,13737,13746,13752,13754,13758,13768,13775,13775,13776,13778,13786,13787,13790,...
    13793,13795,13798,13802,13805,13807,13809,13810,13812,13817,13818,13819,13821,13822,13823,1824,13828,13832,...
    13833,13835,13836,13842,13850,13863,13876,13882,13898,13924,13936,13939,13946,13948,13956,13963,13966,13991,...
    13994,14006,14016,14037,14044,14047,14048,14056,14058,14074,14079,14085,14089,14097,14100,14103,14109,14110,...
    14129,14130,14132,14135,14141,14143,14145,14155,14161,14162,14165,14170,14172,14179,14184,14186,14187,14188,...
    14192,14194,14199,14210,14211,14213,14216,14217,14230,14233,14242,14248,14254,14258,14262,14266,14276,14278,...
    14284,14296,14298,14304,14310,14315,14316,14317,14318,14319,14320,14321,14322,14323,14324,14325,14326,14327,...
    14328,14355,14357,14362,14374,14376,14381,14385,14387,14393,14401,14406,14410,14415,14423,14429,14435,14442,...
    14448,14463,14467,14474,14475,14479,14499,14504,14512,14525,14528,14545,14557,14558,14561,14585,14587,14608,...
    14612,14613,14615,14620,14629,14633,14634,14642,14645,14656,14665,14680,14681,14685,14715,14721,14723,14725,...
    14729,14752,14766,14772,14773,14780,14783,14786,14791,14809,14820,14822,14824,14839,14845,14846,14853,14854,...
    14869,14872,14872,14881,14891,14899,14904,14913,14915,14916,14922,14931,14932,14940,14941,14949,14951,14952,...
    14955,14961,1496514971,14980,14982,15004,15006,15010,15011,15015,15021,15022,15023,15024,15025,15026,15028,...
    15030,15031,15033,15034,15044,15048,15049,15050,15051,15052,15054,15056,15057,15057,15058,15059,15060,15061,...
    15063,15064,15067,15072,15074,15076,15080,15088,15097,15100,15103,15104,15107,15114,15120,15135,15157,15160,...
    15164,15167,15185,15187,15193,15198,15216,15231,15258,15269,15281,15282,15286,15299,15301,15325,15327,15337,...
    15343,15355,15357,15361,15367,15382,15383,15384,15386,15391,15396,15402,15412,15424,15427,15435,15438,15444,...
    15448,15451,15457,15467,15473,15473,15478,15490,15493,15494,15504,15511,15512,15514,15520,15522,15523,15524,...
    15526,15527,15536,15543,15544,15545,15549,15554,15555,15563,15564,15565,15567,15574,15575,15576,15578,15586,...
    15592,15602,15604,15624,15655,15662,15668,15674,15675,15708,15709,15710,157218,15741,15749,15750,15770,15780,...
    15781,15784,15795,15796,15798,15802,15803,15809,15811,15812,1584,15826,15827,15828,15829,15837,15853,15860,...
    15861,15876,15877,15879,15892,15899,15903,15921,15927,15930,15931,15932,15934,15945,15952,15954,15962,16002,...
    16009,16031,16041,16043,16052,16070,16079,16081,16101,16128,16132,16150,16193,16215,16229,16239,16247,16253,...
    16256,16259,16264,16272,16295,16300,16301,16302,16303,16310,16315,16346,16354,16376,16377,16378,16383,16398,...
    16411,16413,16418,16424,16425,16428,16437,16438,16458,16459,16481,16485,16491,16492,16499,16512,16514,16516,...
    16518,16520,16543,16546,16558,16565,16566,16576,16585,16590,16608,16610,16617,16620,16621,16622,16626,16630,...
    16631,16636,16637,16652,16665,16666,16670,16673,16678,16681,16689,16690,16691,16695,16696,16700,16702,16704,...
    16705,16706,16708,16709,16717,16719,16720,16721,16722,16725,16726,16727,16734,16735,16742,16750,16753,16767,...
    16770,16771,16774,16783,16803,16805,16817,16818,16836,16842,16861,16862,16864,16866,16885,16899,16904,16908,...
    16910,16911,16912,16919,16936,16937,16955,16958,16964,16965,16969,16974,16975,16994,16995,17020,17024,17028,...
    17031,17035,17037,17039,17060,17068,17071,17074,17074,17077,17089,17091,17092,17096,17097,17100,17104,17106,...
    17111,17119,17120,17121,17124,17128,17132,17141,17146,17148,17150,17155,17156,17164,17174,17176,17184,17191,...
    17198,17200,17211,17213,17215,17218,17222,17229,17236,17237,17238,17240,17241,17246,17247,17261,17273,17275,...
    17278,17279,17281,17283,17286,17287,17299,17314,17350,17384,17400,17413,17415,17434,17441,17469,17471,17474,...
    17500,17503,17511,17520,17533,17535,17536,17554,17572,17574,17583,17591,17611,17617,17621,17622,17623,17630,...
    17631,17632,17640,17645,17658,17659,17668,17673,17681,17682,17685,17687,17696,17699,17701,17702,17705,17706,...
    17709,17715,17717,17723,17736,17738,17741,17746,17751,17775,17782,17789,17792,17797,17805,17811,17818,17827,...
    17845,17848,17853,17860,17862,17864,17865,17866,17867,17887,17897,17899,17901,17903,17912,17916,17922,17923,...
    17931,17940,17965,17966,17971,17993,17998,18000,18015,18021,18050,18070,1878,18079,18081,18099,18115,18122,...
    18135,18145,18149,18151,18153,18157,18159,18164,18165,18168,18171,18184,18189,18209,18240,18247,18249,18254,...
    18264,18268,18273,18281,18298,18304,18309,18317,18318,18321,18323,18337,18339,18351,18362,18383,18384,18393,...
    18394,18418,18420,18428,18454,18458,18465,18469,18474,18475,18504,18505,18507,18510,18512,18516,18519,18520,...
    18547,18548,18566,18571,18579,18628,18629,18631,18638,18640,18643,18645,18647,18649,18668,18669,18689,18690,...
    18694,18699,18702,18704,18712,18723,18730,18735,18751,18751,18760,18775,18779,18780,18782,18784,18785,18786,...
    18788,18788,18789,18790,18791,18793,18794,18801,18802,18804,18806,18807,18808,18810,18811,18812,18815,18816,...
    18818,18820,18822,18824,18825,18827,18828,18829,18830,18831,18832,18836,18837,18848,18849,18851,18853,18854,...
    18855,18856,18857,18860,18867,18869,18877,18879,18880,18881,18882,18884,18895,18897,18898,18908,18912,18914,...
    18938,18945,18971,18978,18981,18991,18993,19001,19007,19008,19011,19017,19021,19022,19023,19024,19028,19031,...
    19043,19045,19048,19052,19053,19057,19062,19063,19064,19067,19067,19069,19071,19072,19075,19078,19081,19100,...
    19114,19118,19126,19134,19137,19139,19147,19178,19189,19195,19201,19210,19211,19214,19229,19233,19236,19239,...
    19240,19241,19242,19247,19252,19253,19254,19256,19272,19273,19278,19283,19288,19292,19295,19309,19311,19312,...
    19315,19325,19326,19339,19341,19343,19345,19346,19348,19349,19350,19362,19363,19364,19366,19368,19370,19395,...
    19402,19405,19406,19408,19413,19415,19416,19417,19419,19422,19427,19434,19435,19436,19437,19438,19454,19455,...
    19457,19464,19466,19473,19474,19476,19485,19491,19492,19493,19501,19504,19510,19512,19513,19514,19525,19526,...
    19527,19533,19535,19542,19546,19549,19566,19571,19574,19580,19594,19595,19608,19609,19610,19612,19614,19616,...
    19630,19633,19640,19644,19649,19652,19653,19655,19662,19663,19666,19667,19668,19669,19670,19671,19676,19678,...
    19679,19682,19683,19685,19686,19687,19688,19689,19690,19693,19694,19695,19696,19696,19697,19698,19699,19700,...
    19701,19702,19704,19705,19707,19710,19712,19713,19715,19716,19717,19719,19720,19721,19723,19727,19734,19735,...
    19739,19740,19742,19743,19746,19749,19754,19756,19759,19766,19767,19775,19778,19780,19783,19787,19813,19817,...
    19823,19833,19846,19853,19857,19860,19906,19910,19912,19922,19929,19932,19936,19944,19947,19978,19979,19981,...
    19985,19986,19996,19998,20003,20021,20022,20031,20035,20037,20038,20047,20049,20056,20058,20065,20105,20121,...
    20127,20132,20144,20171,20172,20177,20188,20189,20190,20195,20201,20202,20203,20218,20222,20223,20230,20233,...
    20234,20241,20246,20256,20257,20258,20259,20260,20261,20264,20266,20270,20273,20276,20283,20289,20290,20303,...
    20311,20318,20322,20344,20352,20354,20356,20357,20379,20392,20393,20402,20405,20407,20411,20412,20413,20414,...
    20446,20448,20460,20461,20468,20473,20502,20524,20537,20538,20547,20569,20573,20576,20594,20598,20601,20610,...
    20614,20618,20620,20624,20625,20632,20657,20660,20676,20684,20688,20691,20693,20699,20714,20722,20731,20749,...
    20752,20760,20768,20773,20774,20777,20780,20786,20791,20799,20804,20807,20818,20819,20826,20828,20837,20849,...
    20866,20868,20869,20872,20874,20875,20875,20876,20878,20891,20916,20926,20927,20930,20932,20933,20934,20946,...
    20964,20966,20969,20982,20987,20991,20992,20993,20995,20996,21001,21003,21005,21006,21012,21014,21015,21016,...
    21029,21040,21042,21069,21072,21105,21114,21120,21153,21172,21183,21194,21204,21208,21209,21219,21224,21229,...
    21230,21231,21233,21234,21239,21268,21271,21280,21293,21302,21306,21330,21333,21335,21339,21351,21363,21366,...
    21372,21377,21390,21410,21424,21440,21442,21443,21449,21471,21478,21478,21480,21482,21482,21484,21496,...
    25001,25018,25029,25030,25035,25036,25037,25038,25039,25052,25053,25055,25058,25065,25070,25074,25075,25085,...
    25086,25087,25088,25090,25108,25109,25110,25116,25018,25122,25024,25125,25126,25128,25130,25132,25134,25135,...
    25139,25140,25146,25151,25152,25153,25157,25159,25160,25162,25164,25165,25171,25174,25175,25205,25220,25226,...
    25226,25237,25245,25254,25268,25308,25322,25332,25333,25339,25343,25353,25357,25359,25360,25402,25414,25420,...
    25422,25422,25425,25449,25450,25451,25462,25464,25468,25469,25475,25480,25481,25485];

for i=22801:23000
   img=imread(path{1,i});
        figure;
      imshow(img); 
   
end 

% 
%  for i=5001:5030
%     if (num2str(gender(i)) ~='NaN')
%         new_gender(k) = gender(i);
%         path(1,k) = full_path(1,i);
%         img=imread(path{1,k});
%         figure; imshow(img);
%        k=k+1;
%     end
% end   
        

%for i = 1:20
    
   % path = full_path{1,i};
  % siide   img=imread(path); %Read input image
  %   img=rgb2gray(img); % convert to gray
  %  BB=step(faceDetector,img); % Detect faces
  % iimg = insertObjectAnnotation(img, 'rectangle', BB, 'Face'); %Annotate detected faces.
  % figure;
  % imshow(iimg); 
  % title('Detected face');
%end

