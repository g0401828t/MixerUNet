import matplotlib.pyplot as plt
import torch

class plotgraph:
    def __init__(self, loss_list, valloss_list, path, description=""):
        self.path = path
        self.loss_list = loss_list
        self.valloss_list = valloss_list
        self.legend = ['train_loss', "silog", "abs_rel", "log10", "rms", "sq_rel", "log_rms", "d1", "d2", "d3"] 

        # print(min(self.loss_list))
        # print(self.loss_list.index(min(self.loss_list)))
        # print(min(self.valloss_list))
        # print(self.valloss_list.index(min(self.valloss_list)))
        # print(max(self.valacc_list))
        # print(self.valacc_list.index(max(self.valacc_list)))

        print("Saving loss, accuracy graph... ...")
        plt.figure()
        plt.title('model loss')
        plt.subplots(constrained_layout=True)

        # subplot for train loss
        plt.subplot(2, 5, 1)
        plt.plot(self.loss_list)
        plt.title(self.legend[0])
        plt.xlabel('step x 500')
        
        # subplot for val loss
        self.valloss_list = torch.FloatTensor(self.valloss_list)
        self.valloss_list = self.valloss_list.transpose(-1, -2).tolist()
        for i, val_loss in enumerate(self.valloss_list):
                plt.subplot(2, 5, i + 2)
                plt.plot(val_loss)
                plt.title(self.legend[i + 1])
                plt.xlabel('step x 500')
                # plt.ylabel(self.legend[i + 1])

        # plt.legend(['train_loss', "silog", "abs_rel", "log10", "rms", "sq_rel", "log_rms", "d1", "d2", "d3"], loc='upper right')
        plt.savefig(self.path + '/loss_' + description + '.png')
        plt.close("all")